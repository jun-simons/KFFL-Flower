"""Adult (Census Income) dataset loader.

Fetches the Adult dataset from the UCI ML Repository, preprocesses it for
binary classification (income <=50K vs >50K), and returns a standardised
``DatasetBundle`` for consumption by the main dataset handler.

Preprocessing steps:
  1. Drop rows with missing values.
  2. Encode the binary target as 0/1.
  3. Encode each sensitive column as integer codes (kept separate, stacked
     into a 2-D array of shape (n, k)).
  4. One-hot encode all remaining categorical features.
  5. Standardise continuous features to zero mean / unit variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Public contract – every dataset file exposes a ``load() -> DatasetBundle``
# ---------------------------------------------------------------------------


@dataclass
class DatasetBundle:
    """Standardised container returned by every dataset-specific loader.

    Attributes:
        X: Feature matrix, shape ``(n, d)``, float32, already scaled.
            Does *not* contain any sensitive or target column.
        y: Binary target vector, shape ``(n,)``, int64, values in {0, 1}.
        s: Sensitive attribute matrix, shape ``(n, k)``, int64.
            Column ``i`` corresponds to ``sensitive_names[i]``.
        feature_names: Column names for X.
        sensitive_names: Names of the sensitive attributes, length k.
        target_name: Name of the target column.
        sensitive_labels: List of length k; element i maps integer code →
            original string label for sensitive attribute i.
    """

    X: np.ndarray
    y: np.ndarray
    s: np.ndarray                          # shape (n, k)
    feature_names: List[str]
    sensitive_names: List[str]             # length k
    target_name: str
    sensitive_labels: List[Dict[int, str]] # length k


def load(
    sensitive_features: List[str] | None = None,
    target_feature: str = "income",
) -> DatasetBundle:
    """Load and preprocess the Adult dataset.

    Parameters
    ----------
    sensitive_features:
        Columns to treat as sensitive attributes.
        Defaults to ``["sex", "race"]``.
    target_feature:
        Column to treat as the binary label (default ``"income"``).

    Returns
    -------
    DatasetBundle
        ``s`` has shape ``(n, k)`` where ``k = len(sensitive_features)``.
    """
    if sensitive_features is None:
        sensitive_features = ["sex", "race"]

    from ucimlrepo import fetch_ucirepo

    adult = fetch_ucirepo(id=2)  # Adult dataset
    X_raw: pd.DataFrame = adult.data.features.copy()
    y_raw: pd.DataFrame = adult.data.targets.copy()

    df = pd.concat([X_raw, y_raw], axis=1)

    # 1. Drop rows with any missing / "?" values
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. Encode binary target → 0 / 1
    target_col = df[target_feature].astype(str).str.strip().str.rstrip(".")
    unique_targets = sorted(target_col.unique())
    if len(unique_targets) != 2:
        raise ValueError(
            f"Expected binary target, got {len(unique_targets)} classes: {unique_targets}"
        )
    target_map = {unique_targets[0]: 0, unique_targets[1]: 1}
    y = target_col.map(target_map).values.astype(np.int64)

    # 3. Encode each sensitive column → integer codes; stack into (n, k)
    missing = [c for c in sensitive_features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Sensitive feature(s) {missing} not in columns: {list(df.columns)}"
        )

    s_cols: List[np.ndarray] = []
    sensitive_labels: List[Dict[int, str]] = []
    for feat in sensitive_features:
        col = df[feat].astype(str).str.strip()
        categories = sorted(col.unique())
        code_map = {cat: i for i, cat in enumerate(categories)}
        s_cols.append(col.map(code_map).values.astype(np.int64))
        sensitive_labels.append({i: cat for cat, i in code_map.items()})

    s = np.stack(s_cols, axis=1)  # (n, k)

    # 4. Build feature matrix (drop target + all sensitive columns)
    drop_cols = [target_feature] + sensitive_features
    feature_df = df.drop(columns=drop_cols)

    cat_cols = feature_df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    num_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()

    if cat_cols:
        feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)

    # 5. Standardise continuous columns
    feature_names = feature_df.columns.tolist()
    X = feature_df.values.astype(np.float64)
    num_indices = [feature_names.index(c) for c in num_cols if c in feature_names]
    if num_indices:
        X[:, num_indices] = StandardScaler().fit_transform(X[:, num_indices])

    X = X.astype(np.float32)

    return DatasetBundle(
        X=X,
        y=y,
        s=s,
        feature_names=feature_names,
        sensitive_names=list(sensitive_features),
        target_name=target_feature,
        sensitive_labels=sensitive_labels,
    )
