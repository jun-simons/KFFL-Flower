# src/kffl/ml/task.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    lr: float = 0.01
    local_epochs: int = 1
    batch_size: int = 64


def build_model() -> nn.Module:
    # generic example architecture
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))


def get_dataloaders(node_id: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    STUB: Replace with real dataset partitioning.
    For now: deterministic synthetic data so the app runs end-to-end.
    """
    
    seed = (hash(str(node_id)) % 2_000_000_000)  # within 32-bit-ish
    g = torch.Generator().manual_seed(1234 + seed)
    
    x_train = torch.randn(1024, 1, 28, 28, generator=g)
    y_train = torch.randint(0, 10, (1024,), generator=g)
    x_test = torch.randn(256, 1, 28, 28, generator=g)
    y_test = torch.randint(0, 10, (256,), generator=g)

    train = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    return train, test


def train_one_round(model: nn.Module, trainloader: DataLoader, cfg: TrainConfig) -> float:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    last_loss = 0.0

    for _ in range(cfg.local_epochs):
        for xb, yb in trainloader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

    return last_loss
