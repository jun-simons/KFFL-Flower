"""COMPAS (ProPublica; Barenstein, 2019) - two-year recidivism.

Appendix B.2: 6,172 samples after the standard ProPublica filtering, with race
as the protected attribute, evaluated with logistic regression.  The label is
``two_year_recid``, so ``yhat = 1`` is a prediction of recidivism.

Sensitive-attribute variants (config ``sensitive-attribute``):

``african-american``  African-American vs everyone else, K = 2.
``race-sex``          African-American x sex, K = 4 (Sec. 6.5, Figure 5).
``race``              the six recorded races, K = 6 (Sec. 6.5, Figure 6).

The paper reports "52 predictive features" without listing them.  We use the
AIF360 feature set (which FFB, cited for the other datasets, also uses) minus
the 389-level ``c_charge_desc``, and never ``race``, the source of every
variant above.  The paper does not describe a train/test split either; see
``load``.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kffl.dataset.base import Dataset, Split

_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "compas"

CONTINUOUS = [
    "age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count",
]
CATEGORICAL = ["sex", "age_cat", "c_charge_degree"]
RACES = [
    "African-American", "Asian", "Caucasian", "Hispanic", "Native American", "Other",
]

# Number of groups K for each sensitive-attribute variant.
VARIANTS = {"african-american": 2, "race-sex": 4, "race": 6}

TEST_FRACTION = 0.2
SPLIT_SEED = 0


def _read_raw() -> pd.DataFrame:
    path = _DATA_DIR / "compas-scores-two-years.csv"
    if not path.exists():
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_URL, path)
    df = pd.read_csv(path)
    # Standard ProPublica filtering, which yields the 6,172 samples of App. B.2.
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ]
    return df.reset_index(drop=True)


def _groups(df: pd.DataFrame, variant: str) -> np.ndarray:
    african_american = (df["race"] == "African-American").to_numpy(np.int64)
    if variant == "african-american":
        return african_american
    if variant == "race-sex":
        # Footnote 5's four groups, encoded as 2*AA + male:
        # 0 = (other, F), 1 = (other, M), 2 = (AA, F), 3 = (AA, M).
        return 2 * african_american + (df["sex"] == "Male").to_numpy(np.int64)
    return df["race"].map({r: i for i, r in enumerate(RACES)}).to_numpy(np.int64)


def load(
    sensitive_attribute: str = "african-american",
    exclude_sensitive_from_features: bool = True,
) -> Dataset:
    """Load and preprocess COMPAS, caching the encoded arrays to disk.

    Every variant shares one 80/20 split, stratified on race x label so that
    even the 11 Native American samples reach the test set.  The non-IID
    partition always uses the binary African-American indicator, because
    Appendix B.3 defines that split only for a binary group.
    """
    if sensitive_attribute not in VARIANTS:
        raise ValueError(
            f"COMPAS: unsupported sensitive attribute {sensitive_attribute!r}; "
            f"choose from {sorted(VARIANTS)}"
        )

    tag = hashlib.sha1(
        f"{sensitive_attribute}|{exclude_sensitive_from_features}".encode()
    ).hexdigest()[:10]
    cache = _DATA_DIR / f"compas_{tag}.npz"

    if cache.exists():
        z = np.load(cache)
        train = Split(z["Xtr"], z["str"], z["ytr"], z["ptr"])
        test = Split(z["Xte"], z["ste"], z["yte"], z["pte"])
    else:
        df = _read_raw()
        y = df["two_year_recid"].to_numpy(np.float32)
        s = _groups(df, sensitive_attribute)
        s_partition = _groups(df, "african-american")

        idx_tr, idx_te = train_test_split(
            np.arange(len(df)),
            test_size=TEST_FRACTION,
            random_state=SPLIT_SEED,
            stratify=df["race"] + "|" + df["two_year_recid"].astype(str),
        )

        categorical = list(CATEGORICAL)
        if not exclude_sensitive_from_features:
            categorical.append("race")
        elif sensitive_attribute == "race-sex":
            categorical.remove("sex")

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
        X_tr = encoder.fit_transform(df.iloc[idx_tr]).astype(np.float32)
        X_te = encoder.transform(df.iloc[idx_te]).astype(np.float32)

        train = Split(X_tr, s[idx_tr], y[idx_tr], s_partition[idx_tr])
        test = Split(X_te, s[idx_te], y[idx_te], s_partition[idx_te])
        np.savez_compressed(
            cache,
            Xtr=train.X, str=train.s, ytr=train.y, ptr=train.s_partition,
            Xte=test.X, ste=test.s, yte=test.y, pte=test.s_partition,
        )

    return Dataset(
        name="compas",
        task="classification",
        train=train,
        test=test,
        num_features=int(train.X.shape[1]),
        num_groups=VARIANTS[sensitive_attribute],
        sensitive_attribute=sensitive_attribute,
    )
