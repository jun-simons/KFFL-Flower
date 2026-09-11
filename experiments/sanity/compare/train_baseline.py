"""Baseline logistic regression on no_sensitive dataset.

Trains a single logistic regression model (no federation, no proximal penalty)
on the full no_sensitive dataset using Adam, with the same architecture and
roughly the same hyperparameters as the KFFL local update step.

Run from the repo root:
    python experiments/sanity/compare/train_baseline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kffl.data.no_sensitive import load

# ---------------------------------------------------------------------------
# Params (match pyproject.toml)
# ---------------------------------------------------------------------------
LR = 0.01
NUM_EPOCHS = 50   # 10 rounds × 5 local epochs
BATCH_SIZE = 64
SEED = 0

torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
bundle = load(seed=SEED)
X = torch.as_tensor(bundle.X, dtype=torch.float32)
y = torch.as_tensor(bundle.y, dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Label balance: y=0: {(y==0).sum().item()}  y=1: {(y==1).sum().item()}")
print()

# ---------------------------------------------------------------------------
# Model: single linear layer (matches LogisticRegression in kffl/ml/logistic.py)
# ---------------------------------------------------------------------------
model = nn.Linear(X.shape[1], 1)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print(f"{'Epoch':>6}  {'Loss':>8}  {'Acc':>6}")
print("-" * 26)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss, total_n = 0.0, 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch).squeeze(1)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.shape[0]
        total_n += X_batch.shape[0]

    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            logits_all = model(X).squeeze(1)
            epoch_loss = criterion(logits_all, y).item()
            preds = (torch.sigmoid(logits_all) >= 0.5).float()
            acc = (preds == y).float().mean().item()
        print(f"{epoch:>6}  {epoch_loss:>8.4f}  {acc:>6.3f}")

print()
print("Final weight norm:", model.weight.norm().item())
