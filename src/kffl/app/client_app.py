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
    arrays: ArrayRecord = msg.content["arrays"]
    _model = _load_model_from_arrays(arrays)

    # TODO (paper): compute RFF-based local statistics for fairness constraint
    # Placeholder: pretend local_terms is a small tensor/vector
    local_terms = {
        "node_id": int(context.node_id),
        "dummy_stat": float(context.node_id) * 0.1,
    }

    cfg = ConfigRecord({"fair1_blob": dumps(local_terms)})
    content = RecordDict({"fair1": cfg})
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

    # Fairness constraint from server (serialized)
    fairness_blob = config.get("G_blob", None)
    G: Any = loads(fairness_blob) if fairness_blob is not None else None

    # TODO (paper): incorporate G into loss/gradient update
    # For now: normal SGD
    trainloader, _ = get_dataloaders(context.node_id, batch_size=int(config.get("batch_size", 64)))
    train_cfg = TrainConfig(
        lr=float(config.get("lr", 0.01)),
        local_epochs=int(config.get("local_epochs", 1)),
        batch_size=int(config.get("batch_size", 64)),
    )
    train_loss = train_one_round(model, trainloader, train_cfg)

    reply_arrays = ArrayRecord(model.state_dict())
    metrics = MetricRecord(
        {
            "train_loss": float(train_loss),
            "num-examples": len(trainloader.dataset),
            "has_G": 1.0 if G is not None else 0.0,
        }
    )

    content = RecordDict({"arrays": reply_arrays, "metrics": metrics})
    return Message(content=content, reply_to=msg)
