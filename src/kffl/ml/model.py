from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "logreg" # "logreg" | "mlp" | ...
    hidden: int = 128  # used for MLP
    dropout: float = 0.0


def build_model(input_dim: int, cfg: ModelConfig) -> nn.Module:
    if cfg.kind == "logreg":
        return nn.Linear(input_dim, 1)

    if cfg.kind == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

    raise ValueError(f"Unknown model kind: {cfg.kind}")
