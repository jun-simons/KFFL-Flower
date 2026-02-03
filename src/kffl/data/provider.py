from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

_FDS_CACHE: FederatedDataset | None = None

@dataclass
class LocalSplits:
    train: Any
    test: Any

def get_or_create_fds(dataset_name: str, num_partitions: int) -> FederatedDataset:
    global _FDS_CACHE
    if _FDS_CACHE is None:
        _FDS_CACHE = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": IidPartitioner(num_partitions=num_partitions)},
        )
    return _FDS_CACHE

def get_local_splits(
    *,
    context,
    dataset_name: str,
    partition_id: int,
    num_partitions: int,
    test_size: float = 0.2,
    seed: int = 42,
):
    # Cache per-client (split once)
    cache_key = f"splits::{dataset_name}::{partition_id}::{seed}::{test_size}"
    if cache_key in context.state:
        return context.state[cache_key]

    fds = get_or_create_fds(dataset_name, num_partitions)
    part = fds.load_partition(partition_id)

    # Deterministic split
    part_tt = part.train_test_split(test_size=test_size, seed=seed)
    splits = LocalSplits(train=part_tt["train"], test=part_tt["test"])

    context.state[cache_key] = splits
    return splits
