from __future__ import annotations

from typing import Optional, Literal, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def get_local_fs(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
    f_kind: Literal["logits", "probs"] = "logits",
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local (f, s) for FAIR1.

    Assumptions:
      - loader yields tuples: (x_main, s, y)
      - s is already numeric/encoded (n, k)
      - model(x_main) returns logits (n,) or (n,1) for logistic regression,
        or (n,C) for multi-class 

    Returns:
      f: (n, d_f) numpy float32
      s: (n, k)  numpy float32
    """
    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    f_chunks: list[np.ndarray] = []
    s_chunks: list[np.ndarray] = []

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        # adult_collate returns a tuple
        x_main, s, _y = batch

        x_main = x_main.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)

        out = model(x_main)

        # Normalize output shape to (n, d_f)
        if out.ndim == 1:
            out = out.unsqueeze(1) # (n,) -> (n,1)

        if f_kind == "probs":
            # For binary: sigmoid, For multi-class: softmax
            if out.shape[1] == 1:
                out = torch.sigmoid(out)
            else:
                out = torch.softmax(out, dim=1)

        f_chunks.append(out.detach().cpu().numpy().astype(np.float32, copy=False))
        s_chunks.append(s.detach().cpu().numpy().astype(np.float32, copy=False))

    if len(f_chunks) == 0:
        # Empty loader edge case
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )

    f_np = np.concatenate(f_chunks, axis=0)
    s_np = np.concatenate(s_chunks, axis=0)

    return f_np, s_np
