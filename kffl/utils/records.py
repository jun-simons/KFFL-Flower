"""Helpers for moving tensors through Flower ``RecordDict``s."""

from __future__ import annotations

import torch
from flwr.app import Array, ArrayRecord


def state_to_record(state: dict[str, torch.Tensor]) -> ArrayRecord:
    """Serialise a state dict / gradient dict (keys preserve the ordering)."""
    return ArrayRecord({k: Array(v.detach().cpu().numpy()) for k, v in state.items()})


def record_to_state(record: ArrayRecord) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v.numpy()) for k, v in record.items()}


def tensors_to_record(**tensors: torch.Tensor) -> ArrayRecord:
    return ArrayRecord({k: Array(v.detach().cpu().numpy()) for k, v in tensors.items()})
