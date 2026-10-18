"""BANK MARKETING (Moro et al., 2014) - term-deposit subscription.

Appendix B.2: "Data from a Portuguese bank utilized to forecast client
subscriptions to term deposits [...] we consider **age** as the sensitive
attribute", 41,188 samples, as preprocessed by FFB (Han et al., 2023), which
binarises age as ``age >= 25``.  (B.2 also calls the target "Loan Approval";
the dataset's label is the term-deposit subscription ``y``.)

Features follow FFB: every column except ``y`` and ``age``.  That includes
``duration``, the length of the sales call, which is only known after the call
and so leaks the outcome; we keep it for comparability with FFB and the paper.
The paper gives no split; we use one 80/20 split stratified on age group x label.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kffl.dataset.base import Dataset, Split

_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "bank"
_CSV = _DATA_DIR / "bank-additional-full.csv"

CONTINUOUS = [
    "duration", "campaign", "pdays", "previous", "emp.var.rate",
    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
TEST_FRACTION = 0.2
SPLIT_SEED = 0


def _read_raw() -> pd.DataFrame:
    if not _CSV.exists():
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        outer = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(_URL).read()))
        inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
        _CSV.write_bytes(inner.read("bank-additional/bank-additional-full.csv"))
    return pd.read_csv(_CSV, sep=";")


def load(sensitive_attribute: str = "age", exclude_sensitive_from_features: bool = True) -> Dataset:
    """Load and preprocess BANK, caching the encoded arrays to disk."""
    if sensitive_attribute != "age":
        raise ValueError(f"BANK: unsupported sensitive attribute {sensitive_attribute!r}; use 'age'")

    tag = hashlib.sha1(f"{sensitive_attribute}|{exclude_sensitive_from_features}".encode()).hexdigest()[:10]
    cache = _DATA_DIR / f"bank_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        train, test = Split(z["Xtr"], z["str"], z["ytr"]), Split(z["Xte"], z["ste"], z["yte"])
    else:
        df = _read_raw()
        s = (df["age"] >= 25).to_numpy(np.int64)  # FFB: 1 = aged 25 or over
        y = (df["y"] == "yes").to_numpy(np.float32)
        idx_tr, idx_te = train_test_split(
            np.arange(len(df)), test_size=TEST_FRACTION, random_state=SPLIT_SEED,
            stratify=s.astype(str).astype(object) + "|" + y.astype(int).astype(str).astype(object),
        )
        continuous = CONTINUOUS + ([] if exclude_sensitive_from_features else ["age"])
        encoder = ColumnTransformer([
            ("num", StandardScaler(), continuous),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])
        X_tr = encoder.fit_transform(df.iloc[idx_tr]).astype(np.float32)
        X_te = encoder.transform(df.iloc[idx_te]).astype(np.float32)
        train, test = Split(X_tr, s[idx_tr], y[idx_tr]), Split(X_te, s[idx_te], y[idx_te])
        np.savez_compressed(cache, Xtr=train.X, str=train.s, ytr=train.y,
                            Xte=test.X, ste=test.s, yte=test.y)

    return Dataset(name="bank", task="classification", train=train, test=test,
                   num_features=int(train.X.shape[1]), num_groups=2,
                   sensitive_attribute=sensitive_attribute)
