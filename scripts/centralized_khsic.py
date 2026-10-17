"""Centralised reference: minimise the fair objective, Eq. (10), without federation.

    min_w  (1/n) sum_i l(y_i, f(x_i; w))  +  lambda * psi(w)

``psi`` is the KHSIC of Eq. (9), or its label-conditional extension
``sum_y w_y psi_y``.  It is what a converged KFFL run should reach, so it serves
both to validate the federated code and to find the useful range of lambda.

``--kernel exact`` uses the full n_y x n_y kernels (small datasets only);
``--kernel rfm`` uses one fixed draw of D random features, for large n.

    PYTHONPATH=. python scripts/centralized_khsic.py --dataset compas \\
        --sensitive african-american --conditioning label --lambdas 0,3,10,30
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from kffl.dataset import get_dataset
from kffl.fairness.khsic import (
    global_interaction,
    local_terms,
    psi_values,
    strata_onehot,
    stratum_weights,
)
from kffl.fairness.metrics import classification_report
from kffl.fairness.rfm import RandomFeatureMap, derive_seed
from kffl.ml import build_model

MODE = "n-minus-1-sq"  # Eq. (9)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="compas")
    p.add_argument("--sensitive", default="african-american")
    p.add_argument("--conditioning", default="none", choices=["none", "label"])
    p.add_argument("--weights", default="prevalence", choices=["prevalence", "uniform"])
    p.add_argument("--lambdas", default="0,3,10,30,100")
    p.add_argument("--kernel", default="exact", choices=["exact", "rfm"])
    p.add_argument("--rfm-dim", type=int, default=256)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    ds = get_dataset(args.dataset, args.sensitive, True)
    if ds.task != "classification":
        raise ValueError("this reference covers classification only")
    X, y = torch.from_numpy(ds.train.X), torch.from_numpy(ds.train.y)
    X_test = torch.from_numpy(ds.test.X)
    s_enc = torch.nn.functional.one_hot(torch.from_numpy(ds.train.s), ds.num_groups).float()
    R = strata_onehot(y, args.conditioning, ds.num_classes)
    n = R.sum(0)
    w = stratum_weights(n, args.weights)
    bce = torch.nn.BCEWithLogitsLoss()

    if args.kernel == "exact":
        if len(X) > 8000:
            raise ValueError("--kernel exact needs n <= 8000; use --kernel rfm")
        members = [torch.nonzero(R[:, c] > 0).squeeze(-1) for c in range(R.shape[1])]
        centred_Ks = []
        for idx in members:  # H_y K_s^y H_y, constant in w
            K_s = torch.exp(-args.gamma * torch.cdist(s_enc[idx], s_enc[idx]) ** 2)
            centred_Ks.append(K_s - K_s.mean(0, keepdim=True) - K_s.mean(1, keepdim=True) + K_s.mean())

        def psi(model) -> torch.Tensor:
            f = model(X)
            total = 0.0
            for c, idx in enumerate(members):
                K_f = torch.exp(-args.gamma * (f[idx] - f[idx].T) ** 2)
                total = total + w[c] * (centred_Ks[c] * K_f).sum() / (n[c] - 1) ** 2
            return total
    else:
        phi_s = RandomFeatureMap(s_enc.shape[1], args.rfm_dim, args.gamma, derive_seed(0, "s"))
        phi_f = RandomFeatureMap(1, args.rfm_dim, args.gamma, derive_seed(0, "f"))
        Z_s = phi_s(s_enc)

        def psi(model) -> torch.Tensor:
            G, _, _, n_y = global_interaction([local_terms(Z_s, phi_f(model(X)), R)])
            return (w * psi_values(G, n_y, MODE)).sum()

    print(f"{args.dataset}/{args.sensitive}  conditioning={args.conditioning}  "
          f"weights={args.weights}  kernel={args.kernel}  n_y={[int(v) for v in n]}")
    print(" lambda | train BCE |    psi    |  acc   |  SPD    |   EO    | FPR gap | eq.odds | pos_rate", flush=True)
    for lam in [float(v) for v in args.lambdas.split(",")]:
        model = build_model("logistic-regression", X.shape[1], "zeros")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.steps):
            opt.zero_grad()
            loss = bce(model(X).squeeze(-1), y)
            if lam:
                loss = loss + lam * psi(model)
            loss.backward()
            opt.step()
        with torch.no_grad():
            r = classification_report(
                model(X_test).squeeze(-1).numpy().astype(np.float64),
                ds.test.y.astype(np.float64), ds.test.s,
            )
            print(f"{lam:7.1f} | {float(bce(model(X).squeeze(-1), y)):9.4f} | {float(psi(model)):.3e} | "
                  f"{r['accuracy']:.4f} | {r['spd']:+.4f} | {r['eo']:+.4f} | {r['fpr_gap']:+.4f} | "
                  f"{r['eodds']:.4f}  | {r['pos_rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
