"""Dataset registry.  Add a loader here to make it selectable from config."""

from __future__ import annotations

from functools import lru_cache

from kffl.dataset import adult, compas
from kffl.dataset.base import Dataset, Split
from kffl.dataset.partition import partition

LOADERS = {
    "adult": adult.load,
    "compas": compas.load,
}


@lru_cache(maxsize=4)
def get_dataset(
    name: str,
    sensitive_attribute: str,
    exclude_sensitive_from_features: bool,
) -> Dataset:
    """Load a dataset by config name (cached: clients share one process)."""
    try:
        loader = LOADERS[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r}; available: {sorted(LOADERS)}"
        ) from None
    return loader(
        sensitive_attribute=sensitive_attribute,
        exclude_sensitive_from_features=exclude_sensitive_from_features,
    )


__all__ = ["Dataset", "Split", "get_dataset", "partition", "LOADERS"]
