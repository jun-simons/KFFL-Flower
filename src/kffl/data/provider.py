# src/kffl/data/provider.py
from __future__ import annotations
from dataclasses import dataclass

from flwr.common import Context
from flwr_datasets.partitioner import IidPartitioner

from kffl.data.adult import load_adult_client_loaders, AdultClientLoaders


@dataclass(frozen=True)
class DataConfig:
    num_partitions: int
    batch_size: int = 64
    fair_batch_size: int = 512
    seed: int = 42
    # easy to swap partitioners later
    partitioner_factory: callable = lambda n: IidPartitioner(num_partitions=n)


def get_client_loaders(context: Context, cfg: DataConfig) -> AdultClientLoaders:
    partitioner = cfg.partitioner_factory(cfg.num_partitions)
    return load_adult_client_loaders(
        context=context,
        num_partitions=cfg.num_partitions,
        partitioner=partitioner,
        batch_size=cfg.batch_size,
        fair_batch_size=cfg.fair_batch_size,
        seed=cfg.seed,
        # (optional future parameters..)
        # fairness_fraction=1.0,
        # max_fairness_examples=None,
    )
