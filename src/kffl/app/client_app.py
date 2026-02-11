# src/kffl/app/client_app.py
from __future__ import annotations
from typing import Any, Dict

import torch

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp

from kffl.app.message_types import QUERY_FAIR1, TRAIN_KFFL
from kffl.ml.task import TrainConfig, build_model, get_dataloaders, train_one_round
from kffl.utils.serde import dumps, loads
from kffl.fairness.kernel import orf_transform

from kffl.data.provider import get_client_loaders, DataConfig

DATA_CFG = DataConfig(num_partitions=10, batch_size=64, fair_batch_size=512, seed=42)

app = ClientApp()


def _load_model_from_arrays(arrays: ArrayRecord) -> torch.nn.Module:
    model = build_model()
    model.load_state_dict(arrays.to_torch_state_dict())
    return model


@app.query("fair1")
def fair1(msg: Message, context: Context) -> Message:
    """
    FAIR1 (stub):
    - Receive current global model
    - Compute local quantities needed for server to construct global fairness constraint G
    - Reply with serialized local terms
    """
    
    # TODO remove print
    print(f"[CLIENT {context.node_id}] handling query.fair1")
    
    arrays: ArrayRecord = msg.content["arrays"]
    model = _load_model_from_arrays(arrays)

    loaders = get_client_loaders(context, DATA_CFG)
    fairloader = loaders.fairnessloader


    cfg: ConfigRecord = msg.content["cfg"]
    D = int(cfg["D"])
    gamma_s = float(cfg["gamma_s"])
    gamma_f = float(cfg["gamma_f"])
    seed = int(cfg["seed"])

    f, s = get_local_fs(model, split="train", for_fairness=True, context=context)
    n_i = int(len(s))
    if n_i == 0:
        print("No sensitive feature defined in fair1")
        content = RecordDict(
            {
                "fair1": ArrayRecord.from_numpy_ndarrays(
                    [
                        np.zeros((D, D), np.float32),
                        np.zeros((D,), np.float32),
                        np.zeros((D,), np.float32),
                    ]
                ),
                "metrics": ConfigRecord({"num-examples": 0}),
            }
        )
        return Message(content=content, reply_to=msg)

    # --- ORFM feature maps ----
    Zs = orf_transform(s, D=D, gamma=gamma_s, seed=seed)
    Zf = orf_transform(f, D=D, gamma=gamma_f, seed=seed + 1)

    Mi = (Zs.T @ Zf).astype(np.float32) # (D,D)
    mu_s_i = Zs.mean(axis=0).astype(np.float32) #(D,)
    mu_f_i = Zf.mean(axis=0).astype(np.float32) #(D,)

    context.state["Mi"] = ArrayRecord.from_numpy_ndarrays([Mi])

    content = RecordDict(
        {
            "fair1": ArrayRecord.from_numpy_ndarrays([Mi, mu_s_i, mu_f_i]),
            "metrics": ConfigRecord({"num-examples": n_i}),
        }
    )

    return Message(content=content, reply_to=msg)


@app.train("kffl")
def train_kffl(msg: Message, context: Context) -> Message:
    """
    KFFL local update (stub):
    - Receive global model + server-computed fairness constraint G
    - Update locally (you'll replace loss/grad with KFFL objective)
    - Reply with updated weights + metrics
    """
    arrays: ArrayRecord = msg.content["arrays"]
    config: ConfigRecord = msg.content.get("config", ConfigRecord({}))

    model = _load_model_from_arrays(arrays)

    loaders = get_client_loaders(context, DATA_CFG)

    # Fairness constraint from server (serialized)
    fairness_blob = config.get("G_blob", None)
    G: Any = loads(fairness_blob) if fairness_blob is not None else None

    # TODO (paper): incorporate G into loss/gradient update
    # For now: normal SGD
    trainloader = loaders.trainloader
    train_cfg = TrainConfig(
        lr=float(config.get("lr", 0.01)),
        local_epochs=int(config.get("local_epochs", 1)),
        batch_size=int(config.get("batch_size", 64)),
    )
    train_loss = train_one_round(model, trainloader, train_cfg)

    reply_arrays = ArrayRecord(model.state_dict())
    metrics = MetricRecord( 
        {
            "num-examples": len(trainloader.dataset),
            "has_G": 1.0 if G is not None else 0.0,
        }
    )

    content = RecordDict({"arrays": reply_arrays, "metrics": metrics})

    #TODO: test print remove
    print(f"[CLIENT {context.node_id}] handling train.kffl has_G={G is not None}") 
    return Message(content=content, reply_to=msg)
