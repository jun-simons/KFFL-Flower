from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from flwr_datasets import FederatedDataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- Dataset schema (mstz/adult, income) ----
TARGET_COL = "over_threshold"      # int 0/1
SENSITIVE_RACE_COL = "race"        # string
SENSITIVE_SEX_COL = "is_male"      # bool

CONTINUOUS_COLS = ["age", "capital_gain", "capital_loss", "hours_worked_per_week"]
# Everything else (excluding sensitive+target+continuous) treated as categorical for x_main


@dataclass(frozen=True)
class AdultArtifacts:
    x_pre: ColumnTransformer        # builds x_main
    race_pre: OneHotEncoder         # builds race one-hot for s
    x_main_dim: int
    s_dim: int


@dataclass(frozen=True)
class AdultClientLoaders:
    trainloader: DataLoader
    testloader: DataLoader
    fairnessloader: DataLoader
    artifacts: AdultArtifacts


_FDS: Optional[FederatedDataset] = None
_ART: Optional[AdultArtifacts] = None


def _adult_preprocess(dd):
    train = dd["train"]

    def ok(ex: Dict[str, Any]) -> bool:
        # Drop rows with missing values ('?' or None) in any column
        for v in ex.values():
            if v is None:
                return False
            if isinstance(v, str) and v.strip() == "?":
                return False
        return True

    train = train.filter(ok)
    return {"train": train}


def adult_collate(examples: list[dict[str, Any]], art: AdultArtifacts) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate a list of HF examples into (x_main, s, y) tensors."""
    # HF examples are dicts of scalars/strings; convert to dict-of-lists
    batch: dict[str, list[Any]] = {}
    for ex in examples:
        for k, v in ex.items():
            batch.setdefault(k, []).append(v)

    t = _batch_to_tensors(batch, art)
    return t["x_main"], t["s"], t["y"]


def _get_fds(partitioner) -> FederatedDataset:
    global _FDS
    if _FDS is not None:
        return _FDS

    _FDS = FederatedDataset(
        dataset="mstz/adult",
        subset="income",
        partitioners={"train": partitioner},
        preprocessor=_adult_preprocess,
        shuffle=True,
        seed=42,
    )
    return _FDS


def _fit_artifacts_on_sample(train_part, max_rows: int = 50_000) -> AdultArtifacts:
    n = len(train_part)
    take = min(n, max_rows)
    sample = train_part.select(range(take))

    all_cols = list(sample.features.keys())

    # Categorical columns for x_main:
    # everything that's not continuous/sensitive/target
    cat_cols = [
        c for c in all_cols
        if c not in set(CONTINUOUS_COLS)
        and c not in {SENSITIVE_RACE_COL, SENSITIVE_SEX_COL, TARGET_COL}
    ]

    x_pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    df = sample.to_pandas()

    # Ensure types
    for c in CONTINUOUS_COLS:
        df[c] = df[c].astype(np.float32)

    for c in cat_cols:
        df[c] = df[c].astype(str)

    x_pre.fit(df[CONTINUOUS_COLS + cat_cols])
    
    race_pre = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    race_pre.fit(df[[SENSITIVE_RACE_COL]].astype(str))

    x_main_dim = int(x_pre.transform(df[CONTINUOUS_COLS + cat_cols]).shape[1])
    s_dim = int(race_pre.transform(df[[SENSITIVE_RACE_COL]].astype(str)).shape[1]) + 1
    return AdultArtifacts(x_pre=x_pre, race_pre=race_pre, x_main_dim=x_main_dim, s_dim=s_dim)


def _batch_to_tensors(batch: Dict[str, Any], art: AdultArtifacts) -> Dict[str, torch.Tensor]:
    # x_main input dict: continuous + all non-sensitive, non-target, non-continuous columns as strings
   
    # Build a small DF for sklearn
    df = {}
    for c in CONTINUOUS_COLS:
        df[c] = np.asarray(batch[c], dtype=np.float32)

    for c in batch.keys():
        if c in CONTINUOUS_COLS or c in {SENSITIVE_RACE_COL, SENSITIVE_SEX_COL, TARGET_COL}:
            continue
        df[c] = np.asarray(batch[c]).astype(str)

    df = pd.DataFrame(df)

    x_main = art.x_pre.transform(df).astype(np.float32)
        # s = [race_onehot, is_male_float]
   
    race_df = pd.DataFrame({SENSITIVE_RACE_COL: np.asarray(batch[SENSITIVE_RACE_COL]).astype(str)})
    race_oh = art.race_pre.transform(race_df).astype(np.float32)

    is_male = np.asarray(batch[SENSITIVE_SEX_COL], dtype=np.float32).reshape(-1, 1)  # bool -> 0/1 float
    s = np.concatenate([race_oh, is_male], axis=1).astype(np.float32)

    y = np.asarray(batch[TARGET_COL], dtype=np.float32).reshape(-1)  # 0/1

    return {
        "x_main": torch.from_numpy(x_main),
        "s": torch.from_numpy(s),
        "y": torch.from_numpy(y),
    }


def load_adult_client_loaders(
    *,
    context,
    num_partitions: int,
    partitioner,
    batch_size: int = 128,
    fair_batch_size: int = 512,
    seed: int = 42,
) -> AdultClientLoaders:
    pid = int(context.node_config["partition-id"])
    cache_key = f"adult::{pid}::{batch_size}::{fair_batch_size}::{seed}"

    if cache_key in context.state:
        return context.state[cache_key]

    fds = _get_fds(partitioner=partitioner)
    part = fds.load_partition(pid, split="train")

    global _ART
    if _ART is None:
        _ART = _fit_artifacts_on_sample(part)
    art = _ART
    
    tt = part.train_test_split(test_size=0.2, seed=seed)

    train_ds = tt["train"].flatten_indices()
    test_ds  = tt["test"].flatten_indices()

    trainloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: adult_collate(batch, art),
    )

    testloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: adult_collate(batch, art),
    )
    fairnessloader = DataLoader(
        train_ds,
        batch_size=fair_batch_size,
        shuffle=False,
        collate_fn=lambda batch: adult_collate(batch, art),
    )
    
    out = AdultClientLoaders(trainloader=trainloader, testloader=testloader, fairnessloader=fairnessloader, artifacts=_ART)
    context.state[cache_key] = out
    return out
