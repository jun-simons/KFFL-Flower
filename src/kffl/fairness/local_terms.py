from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class FSConfig:
    # If set, only use up to this many examples for fairness stats (useful later?)
    max_examples: Optional[int] = None
    # Device override (None => infer from model)
    device: Optional[torch.device] = None


def get_local_fs(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    cfg: FSConfig = FSConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      f: model outputs for each sample, shape (N, d_out) or (N,)
      s: sensitive features, shape (N, k)

    Assumes each loader batch is (x_main, s, y) OR (x_main, s) OR a dict with keys.
    """
    model.eval()
    dev = cfg.device
    if dev is None:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")

    f_chunks: list[np.ndarray] = []
    s_chunks: list[np.ndarray] = []

    seen = 0
    with torch.no_grad():
        for batch in loader:
            # Support tuple batches: (x_main, s, y) or (x_main, s)
            if isinstance(batch, (tuple, list)):
                x_main = batch[0]
                s = batch[1]
            else:
                # dict-like batch: {"x_main":..., "s":...}
                x_main = batch["x_main"]
                s = batch["s"]

            x_main = x_main.to(dev)
            out = model(x_main)

            # Make sure output is at least 2D for consistent concatenation
            # (N,) -> (N,1)
            if out.ndim == 1:
                out = out.unsqueeze(1)

            # Move to CPU numpy
            f_np = out.detach().cpu().numpy()
            s_np = s.detach().cpu().numpy()

            # Optional truncation for speed/experiments
            if cfg.max_examples is not None:
                remaining = cfg.max_examples - seen
                if remaining <= 0:
                    break
                if f_np.shape[0] > remaining:
                    f_np = f_np[:remaining]
                    s_np = s_np[:remaining]

            f_chunks.append(f_np.astype(np.float32, copy=False))
            s_chunks.append(s_np.astype(np.float32, copy=False))
            seen += f_np.shape[0]

            if cfg.max_examples is not None and seen >= cfg.max_examples:
                break

    if not f_chunks:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    f = np.concatenate(f_chunks, axis=0)
    s = np.concatenate(s_chunks, axis=0)
    return f, s
