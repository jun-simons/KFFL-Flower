"""ADULT (Becker & Kohavi, 1996) - binary income classification.

Appendix B.2: "up to 14 attributes used in predicting whether an individual
would earn an income >= 50K or <= 50K [...] we consider **sex** as the
protected sensitive variable", evaluated with logistic regression.

The canonical UCI train/test split is used (n = 32,561 / 16,281), matching the
"n = 32K" quoted in Section 4.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kffl.dataset.base import Dataset, Split

_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "adult"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]
CONTINUOUS = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week",
]
CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]

# Sensitive-group encoding.  SPD = Pr(yhat=1 | s=1) - Pr(yhat=1 | s=0) (Eq. 1),
# so mapping Male -> 1 reproduces the paper's positive FedAvg SPD (~0.18).
SENSITIVE_LEVELS = {"sex": {"Female": 0, "Male": 1}}


def _download(name: str) -> Path:
    path = _DATA_DIR / name
    if not path.exists():
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_BASE_URL + name, path)
    return path


def _read_raw(name: str, skiprows: int) -> pd.DataFrame:
    df = pd.read_csv(
        _download(name),
        header=None,
        names=COLUMNS,
        skiprows=skiprows,
        skipinitialspace=True,
        skip_blank_lines=True,
        # NOTE: "?" is kept as an explicit category rather than dropped, so that
        # n stays at the 32,561 quoted in the paper (dropping would give 30,162).
        keep_default_na=False,
    )
    # adult.test writes the label with a trailing period ("<=50K.").
    df["income"] = df["income"].str.rstrip(".").str.strip()
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()
    return df


def load(
    sensitive_attribute: str = "sex",
    exclude_sensitive_from_features: bool = True,
) -> Dataset:
    """Load and preprocess ADULT, caching the encoded arrays to disk.

    The encoder/scaler is fitted on the training split and applied to the test
    split.  The paper does not describe its preprocessing; this is the standard
    recipe and is shared by every client so that all ``X`` live in one space.
    """
    if sensitive_attribute not in SENSITIVE_LEVELS:
        raise ValueError(
            f"ADULT: unsupported sensitive attribute {sensitive_attribute!r}; "
            f"choose from {sorted(SENSITIVE_LEVELS)}"
        )

    tag = hashlib.sha1(
        f"{sensitive_attribute}|{exclude_sensitive_from_features}".encode()
    ).hexdigest()[:10]
    cache = _DATA_DIR / f"adult_{tag}.npz"

    if cache.exists():
        z = np.load(cache)
        train = Split(z["Xtr"], z["str"], z["ytr"])
        test = Split(z["Xte"], z["ste"], z["yte"])
    else:
        df_tr = _read_raw("adult.data", skiprows=0)
        df_te = _read_raw("adult.test", skiprows=1)  # line 1 is a stray comment

        levels = SENSITIVE_LEVELS[sensitive_attribute]
        s_tr = df_tr[sensitive_attribute].map(levels).to_numpy(np.int64)
        s_te = df_te[sensitive_attribute].map(levels).to_numpy(np.int64)

        y_tr = (df_tr["income"] == ">50K").to_numpy(np.float32)
        y_te = (df_te["income"] == ">50K").to_numpy(np.float32)

        categorical = list(CATEGORICAL)
        if exclude_sensitive_from_features:
            # App. F: "we exclude the sensitive attribute s_i from the training
            # process".  Applied to classification too, for consistency.
            categorical.remove(sensitive_attribute)

        encoder = ColumnTransformer(
            [
                ("num", StandardScaler(), CONTINUOUS),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical,
                ),
            ]
        )
        X_tr = encoder.fit_transform(df_tr).astype(np.float32)
        X_te = encoder.transform(df_te).astype(np.float32)

        train, test = Split(X_tr, s_tr, y_tr), Split(X_te, s_te, y_te)
        np.savez_compressed(
            cache,
            Xtr=X_tr, str=s_tr, ytr=y_tr,
            Xte=X_te, ste=s_te, yte=y_te,
        )

    return Dataset(
        name="adult",
        task="classification",
        train=train,
        test=test,
        num_features=int(train.X.shape[1]),
        num_groups=len(SENSITIVE_LEVELS[sensitive_attribute]),
        sensitive_attribute=sensitive_attribute,
    )
