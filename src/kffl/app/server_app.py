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
        fair1_msgs: list[Message] = []   # <-- this line must exist before append

        for nid in selected:
           rd = RecordDict({"arrays": global_arrays})
           fair1_msgs.append(
               Message(
                   content=rd,
                   dst_node_id=nid,
                   message_type=QUERY_FAIR1,
                   group_id=str(server_round),
               )
           )
        
        fair1_replies = list(grid.send_and_receive(fair1_msgs))
        local_terms = []
        for rep in fair1_replies:
            if rep.has_content() and "fair1" in rep.content:
                cfg: ConfigRecord = rep.content["fair1"]
                local_terms.append(loads(cfg["fair1_blob"]))

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

        for rep in train_replies:
            if not rep.has_content():
                continue

            arrays: ArrayRecord = rep.content["arrays"]
            metrics = rep.content.get("metrics", None)

            n = int(metrics["num-examples"]) if metrics is not None and "num-examples" in metrics else 1
            client_weights.append(n)

            client_sds.append(arrays.to_torch_state_dict())

        if client_sds:
            avg_sd = fedavg_state_dict(client_sds, client_weights)
            global_arrays = ArrayRecord.from_torch_state_dict(avg_sd)

        print(f"[ROUND {server_round}] selected={len(selected)} G={G}")   
