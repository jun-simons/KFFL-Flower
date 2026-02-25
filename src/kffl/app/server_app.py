# src/kffl/app/server_app.py
from __future__ import annotations
#from typing import Dict, Iterable, List, Tuple

import random
import numpy as np
import torch

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp

from kffl.app.message_types import QUERY_FAIR1, TRAIN_KFFL
from kffl.ml.task import build_model
from kffl.utils.serde import dumps, loads

app = ServerApp()


def _sample_nodes(node_ids: list[int], fraction: float, min_nodes: int) -> list[int]:
    n = max(min_nodes, int(round(len(node_ids) * fraction)))
    n = min(n, len(node_ids))
    return random.sample(node_ids, n)

def _fedavg_ndarrays(ndarrays_list: list[list[np.ndarray]], weights: list[int]) -> list[np.ndarray]:
    """FedAvg over a list-of-lists of NumPy ndarrays."""
    total = float(sum(weights))
    avg = [np.zeros_like(arr) for arr in ndarrays_list[0]]

    for client_params, w in zip(ndarrays_list, weights):
        scale = w / total
        for j, arr in enumerate(client_params):
            avg[j] += arr * scale

    return avg


def fedavg_state_dict(
    sds: list[dict[str, torch.Tensor]],
    weights: list[int],
) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    out: dict[str, torch.Tensor] = {}

    for k in sds[0].keys():
        acc = None
        for sd, w in zip(sds, weights):
            t = sd[k].detach().cpu()
            acc = t * (w / total) if acc is None else acc + t * (w / total)
        out[k] = acc
    return out

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read config from pyproject.toml
    num_rounds = int(context.run_config.get("num-server-rounds", 3))
    fraction_train = float(context.run_config.get("fraction-train", 1.0))
    min_nodes = int(context.run_config.get("min-nodes", 2))

    lr = float(context.run_config.get("lr", 0.01))
    local_epochs = int(context.run_config.get("local-epochs", 1))
    batch_size = int(context.run_config.get("batch-size", 64))

    # Init global model
    model = build_model()
    global_arrays = ArrayRecord(model.state_dict())

    all_nodes = list(grid.get_node_ids())

    for server_round in range(1, num_rounds + 1):
        selected = _sample_nodes(all_nodes, fraction_train, min_nodes)

        # ---- (1) FAIR1: query local terms ----
        fair1_msgs: list[Message] = []

        D = 1024
        gamma_s = 0.5
        gamma_f = 0.5
        seed = 42

        fair1_msgs: list[Message] = []
        for nid in selected:
            rd = RecordDict(
                {
                    "arrays": global_arrays, # model params
                    "cfg": ConfigRecord( # RFM kernal parameters
                        {"D": D, "gamma_s": gamma_s, "gamma_f": gamma_f, "seed": seed}
                    )
                }
            )
            fair1_msgs.append(
                Message(
                    content=rd,
                    dst_node_id=nid,
                    message_type=QUERY_FAIR1,
                    group_id=str(server_round),
                )
            )

        fair1_replies = list(grid.send_and_receive(fair1_msgs))

        M_sum = np.zeros((D,D), dtype=np.float32)
        mu_s_sum = np.zeros((D,), dtype=np.float32)
        mu_f_sum = np.zeros((D,), dtype=np.float32)
        n_total = 0
        
        for rep in fair1_replies:
            if not rep.has_content():
                continue
            if "fair1" not in rep.content:
                continue

            fair_arr: ArrayRecord = rep.content["fair1"]
            Mi, mu_s_i, mu_f_i = fair_add.to_numpy_ndarrays()

            metrics = rep.content.get("metrics", None)
            n_i = int(metrics["num-examples"])

            Mi = Mi.astype(np.float32, copy=False)
            mu_s_i = mu_s_i.astype(np.float32, copy=False)
            mu_f_i = mu_f_i.astype(np.float32, copy=False)

            M_sum += Mi
            mu_s_sum += n_i * mu_s_i
            mu_f_sum += n_i * mu_f_i
            n_total += n_i

        mu_s = mu_s_sum / max(n_total, 1)
        mu_f = mu_f_sum / max(n_total, 1)
        G = M_sum - float(n_total) * np.outer(mu_s, mu_f).astype(np.float32)
        
        # TODO test print remove
        G_blob = {"D": D, "gamma_s": gamma_s, "gamma_f": gamma_f, "seed": seed} 
        print(f"[ROUND {server_round}] FAIR1 n_total={n_total} ||G||={float(np.linalg.norm(G)):.4f}")
            
        # TODO (paper): compute global fairness constraint G from local FAIR1 terms
        # Placeholder "G": just store how many clients contributed
        G = {"round": server_round, "num_clients": len(local_terms)}

        # ---- (2) TRAIN: send G + hyperparams; receive updated weights ----
        train_cfg = ConfigRecord(
            {
                "server_round": server_round,
                "lr": lr,
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "G_blob": dumps(G),
            }
        )

        train_msgs: list[Message] = []
        for nid in selected:
            rd = RecordDict({"arrays": global_arrays, "config": train_cfg})
            
            train_msgs.append( 
                Message( content=rd,
                    dst_node_id=nid,
                    message_type=TRAIN_KFFL,
                    group_id=str(server_round),
                )
            )


        train_replies = list(grid.send_and_receive(train_msgs))

        # ---- (3) Aggregate (FedAvg) ----
        client_sds: list[dict[str, torch.Tensor]] = []
        client_weights: list[int] = []

        total_examples = 0
        weighted_loss_sum = 0.0
        num_with_loss = 0

        for rep in train_replies:
            if not rep.has_content():
                continue

            arrays: ArrayRecord = rep.content["arrays"]
            metrics = rep.content.get("metrics", None)

            n = int(metrics["num-examples"]) if metrics is not None and "num-examples" in metrics else 1
            client_weights.append(n)

            loss = float(metrics["train_loss"])
            total_examples += n
            weighted_loss_sum += n * loss
            num_with_loss += 1

            client_sds.append(arrays.to_torch_state_dict())

        if client_sds:
            avg_sd = fedavg_state_dict(client_sds, client_weights)
            global_arrays = ArrayRecord.from_torch_state_dict(avg_sd)
        
        w0 = next(iter(global_arrays.to_torch_state_dict().values()))
        print(f"[ROUND {server_round}] w0_norm={float(w0.norm()):.4f} G={G}")
        avg_train_loss = (weighted_loss_sum / total_examples) if total_examples > 0 else float("nan")
        print(f"[ROUND {server_round}] avg_train_loss={avg_train_loss:.4f} reports={num_with_loss}/{len(train_replies)}")
        print(f"[ROUND {server_round}] selected={len(selected)} G={G}")   
