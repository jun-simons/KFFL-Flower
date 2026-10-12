"""KFFL - Server Side (Algorithm 3).

each global iteration ``t`` includes three communication rounds:

    FAIR1  down w_t (+ zeta)              up Phi^i(w_t) = {M^i, mu_{s,i}, mu_{f,i}}
           server: G(w_t) (13), mu_s (17), mu_f (18)
    FAIR2  down G(w_t), mu_s, mu_f        up g^i(w_t)              (Eqs. 15, 16)
           server: w_{t+1/2} = w_t - alpha * lambda * sum_i g^i    (Alg. 1, line 3)
    UPDATE down w_{t+1/2}                 up w^i_{t+1}             (Alg. 1, line 5)
           server: w_{t+1} = average(w^i_{t+1})                    (Alg. 1, line 7)

Uplink and downlink are therefore``2|w| + D^2`` each, matching Tabel 2

``algorithm = "fedavg"`` skips FAIR1/FAIR2 for the unregularised baseline
"""

from __future__ import annotations

import json
import time
from logging import INFO
from pathlib import Path

import numpy as np
import torch
from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

from kffl.config import KFFLConfig
from kffl.dataset import get_dataset
from kffl.fairness.khsic import (
    LocalTerms,
    global_interaction,
    psi_values,
    stratum_scales,
    stratum_weights,
)
from kffl.fairness.metrics import classification_report, regression_report
from kffl.ml import build_model
from kffl.utils import record_to_state, state_to_record, tensors_to_record

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = KFFLConfig.from_run_config(context.run_config)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    ds = get_dataset(
        cfg.dataset, cfg.sensitive_attribute, cfg.exclude_sensitive_from_features
    )
    model = build_model(cfg.model, ds.num_features, cfg.model_init)
    weights = {k: v.clone() for k, v in model.state_dict().items()}  # w_0

    # App. G: with rfm-resample="once" the features are drawn a single time and
    # held fixed; with "per-round" a fresh zeta is broadcast each iteration, as
    # Algorithm 3 line 5 does (and as Corollary 1's unbiasedness requires).
    fixed_zeta = int(rng.integers(2**31 - 1))

    history: list[dict] = []
    log(INFO, "KFFL config: %s", _summary(cfg, ds))

    for t in range(cfg.num_server_rounds):
        started = time.time()
        node_ids = list(grid.get_node_ids())
        record: dict[str, float] = {"round": t + 1, "num_clients": len(node_ids)}

        w_half = weights
        if cfg.uses_fairness and cfg.fairness_lambda != 0.0:
            zeta = fixed_zeta if cfg.rfm_resample == "once" else int(rng.integers(2**31 - 1))

            # ---- FAIR1: collect Phi^i(w_t), assemble G(w_t) ------------------
            terms = _fair1(grid, node_ids, weights, zeta, t)
            G, mu_s, mu_f, n = global_interaction(terms)  # one G_y per stratum
            psi_y = psi_values(G, n, cfg.psi_normalization)
            w_y = stratum_weights(n, cfg.khsic_stratum_weights)
            record["psi"] = float((w_y * psi_y).sum())
            if len(psi_y) > 1:
                record.update({f"psi_y{c}": float(v) for c, v in enumerate(psi_y)})
            record["n"] = int(n.sum())

            # ---- FAIR2: collect g^i(w_t), take the fairness step -------------
            # The constants of Eq. (9) / App. A.6 and the weights w_y travel to
            # the clients as c_y, so the summed reply is grad psi itself.
            scale = stratum_scales(
                n, cfg.psi_normalization, cfg.khsic_stratum_weights
            ).to(G.dtype)
            g = _fair2(grid, node_ids, G, mu_s, mu_f, scale, t)
            # Alg. 1 line 3, with lambda from Eq. (10) reinstated (Sec. 3 absorbs
            # it into psi).
            step = cfg.fairness_step_size * cfg.fairness_lambda
            w_half = {k: weights[k] - step * g[k] for k in weights}
            record["fair_grad_norm"] = float(  # ||grad psi||
                torch.sqrt(sum((g[k] ** 2).sum() for k in g))
            )
            record["fair_step_norm"] = step * record["fair_grad_norm"]

        # ---- Local Update: prox step, then aggregate -------------------------
        prox_mu = cfg.prox_mu if cfg.uses_fairness else 0.0
        locals_, sizes, losses = _local_update(grid, node_ids, w_half, prox_mu, t)
        weights = _aggregate(locals_, sizes, cfg.aggregation)  # Alg. 1, line 7
        record["train_loss"] = float(np.mean(losses)) if losses else float("nan")

        # ---- Central evaluation on the held-out test split --------------------
        record.update(_evaluate(model, weights, ds))
        record["seconds"] = time.time() - started
        history.append(record)
        log(INFO, "round %2d/%d  %s", t + 1, cfg.num_server_rounds, _fmt(record))

    _save(cfg, ds, history)


# --------------------------------------------------------------------------
# Sub-rounds
# --------------------------------------------------------------------------


def _fair1(
    grid: Grid, node_ids: list[int], weights: dict, zeta: int, t: int
) -> list[LocalTerms]:
    """Algorithm 3, lines 4-8: gather ``Phi(w_t) = {Phi^i(w_t)}``."""
    def content() -> RecordDict:
        return RecordDict(
            {
                "arrays": state_to_record(weights),
                "config": ConfigRecord({"zeta": zeta, "round": t}),
            }
        )

    replies = grid.send_and_receive(
        [
            Message(content=content(), message_type="query.fair1", dst_node_id=nid,
                    group_id=str(t))
            for nid in node_ids
        ]
    )
    terms = []
    for reply in replies:
        _check(reply, "FAIR1")
        arrays = reply.content["arrays"]
        terms.append(
            LocalTerms(
                M=torch.from_numpy(arrays["M"].numpy()),
                mu_s=torch.from_numpy(arrays["mu_s"].numpy()),
                mu_f=torch.from_numpy(arrays["mu_f"].numpy()),
                n=torch.from_numpy(arrays["n"].numpy()),
            )
        )
    return terms


def _fair2(
    grid: Grid,
    node_ids: list[int],
    G: torch.Tensor,
    mu_s: torch.Tensor,
    mu_f: torch.Tensor,
    scale: torch.Tensor,
    t: int,
) -> dict[str, torch.Tensor]:
    """Algorithm 3, lines 10-13: gather ``g^i(w_t)`` and sum them (Lemma 2).

    Algorithm 3 line 11 ships the single matrix ``mu_s mu_f^T``; we send the two
    D-vectors instead (2D floats vs D^2, so the accounting of Table 2 is
    unaffected) because the exact Jacobian of Eq. (15) needs ``mu_s`` alone.
    """
    replies = grid.send_and_receive(
        [
            Message(
                content=RecordDict(
                    {
                        "arrays": tensors_to_record(
                            G=G, mu_s=mu_s, mu_f=mu_f, scale=scale
                        )
                    }
                ),
                message_type="query.fair2",
                dst_node_id=nid,
                group_id=str(t),
            )
            for nid in node_ids
        ]
    )
    total: dict[str, torch.Tensor] = {}
    for reply in replies:
        _check(reply, "FAIR2")
        for key, value in record_to_state(reply.content["arrays"]).items():
            total[key] = value if key not in total else total[key] + value
    return total


def _local_update(
    grid: Grid, node_ids: list[int], w_half: dict, prox_mu: float, t: int
) -> tuple[list[dict], list[int], list[float]]:
    """Algorithm 3, lines 15-17."""
    def content() -> RecordDict:
        return RecordDict(
            {
                "arrays": state_to_record(w_half),
                "config": ConfigRecord({"prox_mu": prox_mu, "round": t}),
            }
        )

    replies = grid.send_and_receive(
        [
            Message(content=content(), message_type="train.local_update",
                    dst_node_id=nid, group_id=str(t))
            for nid in node_ids
        ]
    )
    models, sizes, losses = [], [], []
    for reply in replies:
        _check(reply, "Local Update")
        models.append(record_to_state(reply.content["arrays"]))
        sizes.append(int(reply.content["metrics"]["n_i"]))
        losses.append(float(reply.content["metrics"]["loss"]))
    return models, sizes, losses


def _aggregate(
    models: list[dict], sizes: list[int], mode: str
) -> dict[str, torch.Tensor]:
    """``w_{t+1} = average(w^i_{t+1})``.

    Algorithm 1 line 7 specifies the uniform mean ``(1/m) sum_i w^i``; Algorithm
    5 line 25 instead says "FedAvg", i.e. the n_i-weighted mean.  They agree in
    the IID setting, where all clients hold the same number of samples.
    """
    if mode == "uniform":
        w = np.ones(len(models)) / len(models)
    elif mode == "weighted":
        w = np.asarray(sizes, dtype=float) / float(sum(sizes))
    else:
        raise ValueError(f"unknown aggregation {mode!r}")
    return {
        key: sum(float(wi) * m[key] for wi, m in zip(w, models)) for key in models[0]
    }


# --------------------------------------------------------------------------
# Evaluation and bookkeeping
# --------------------------------------------------------------------------


def _evaluate(model, weights: dict, ds) -> dict[str, float]:
    """Central evaluation on the test split, as the paper reports it."""
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(ds.test.X)).squeeze(-1).numpy()
    report = (
        classification_report if ds.task == "classification" else regression_report
    )
    return report(scores.astype(np.float64), ds.test.y.astype(np.float64), ds.test.s)


def _check(reply: Message, stage: str) -> None:
    if not reply.has_content():
        raise RuntimeError(f"{stage}: client {reply.metadata.src_node_id} failed: "
                           f"{reply.error}")


def _summary(cfg: KFFLConfig, ds) -> str:
    return (
        f"{cfg.algorithm} | {ds.name}/{cfg.model} | {cfg.partition} | "
        f"khsic={cfg.khsic_conditioning}/{cfg.khsic_stratum_weights} "
        f"lambda={cfg.fairness_lambda} D={cfg.rfm_dim} alpha={cfg.alpha} "
        f"prox_mu={cfg.prox_mu:.4g} rounds={cfg.num_server_rounds} seed={cfg.seed}"
    )


def _fmt(record: dict) -> str:
    keys = ["accuracy", "spd", "eo", "rmse", "ks", "pos_rate", "train_loss", "psi"]
    if record.get("spd") != record.get("spd"):  # NaN: K > 2, show the multi-group gaps
        keys[1:3] = ["spd_multi", "eo_multi"]
    return "  ".join(
        f"{k}={record[k]:.4f}" for k in keys if k in record and record[k] == record[k]
    )


def _save(cfg: KFFLConfig, ds, history: list[dict]) -> None:
    path = Path(cfg.results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {k: v for k, v in cfg.__dict__.items()},
        "dataset": {"name": ds.name, "n_train": len(ds.train), "n_test": len(ds.test),
                    "num_features": ds.num_features, "num_groups": ds.num_groups},
        "history": history,
        "final": history[-1] if history else {},
    }
    path.write_text(json.dumps(payload, indent=2))
    log(INFO, "wrote %s", path)
