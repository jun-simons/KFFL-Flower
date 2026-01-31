# src/kffl/utils/serde.py

import pickle
from typing import Any

def dumps(obj: Any) -> bytes:
    # serialize non-metric non-weights objects to ship in ConfigRecord
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

def loads(blob: bytes) -> Any:
    return pickle.loads(blob)
