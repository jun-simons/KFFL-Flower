"""Tests for the KFFL 3-round message protocol.

Exercises client-side handlers in isolation with constructed Message objects.
FAIR1 and FAIR2 tests monkeypatch ``_load_partition`` with a synthetic
DataLoader so no UCI ML repository download is required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from flwr.app import Array, ArrayRecord, ConfigRecord, Message, RecordDict
from kffl.ml import LogisticRegression, get_weights


# ---------------------------------------------------------------------------
# Stub Context
# ---------------------------------------------------------------------------


class _StubContext:
    def __init__(self, node_id: int = 0, model_size: int = 16) -> None:
        self.node_id = node_id
        self.run_config = {"num-random-features": D, "model-size": model_size}
        self.node_config = {"partition-id": 0}


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

D = 8
INPUT_DIM = 10
NUM_CLASSES = 2
MODEL_SIZE = 16
N_SAMPLES = 60
K_SENSITIVE = 2


# ---------------------------------------------------------------------------
# Synthetic DataLoader fixture
# ---------------------------------------------------------------------------


def _synthetic_loader(
    n: int = N_SAMPLES,
    input_dim: int = INPUT_DIM,
    k: int = K_SENSITIVE,
) -> DataLoader:
    rng = torch.Generator().manual_seed(0)
    X = torch.randn(n, input_dim, generator=rng)
    y = (X[:, 0] > 0).long()
    s = torch.randint(0, 2, (n, k), generator=rng).long()
    return DataLoader(TensorDataset(X, y, s), batch_size=32, shuffle=False)


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def _model_weights_arec(input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES) -> ArrayRecord:
    model = LogisticRegression(input_dim=input_dim, num_classes=num_classes)
    return ArrayRecord({f"w_{i}": Array(w) for i, w in enumerate(get_weights(model))})


def _model_config(seed: int = 42, D: int = D) -> ConfigRecord:
    return ConfigRecord({
        "round_num": 0,
        "seed": seed,
        "num_rf": D,
        "gamma_s": 1.0,
        "gamma_f": 1.0,
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
        "model_name": "logistic",
    })


def _make_fair1_message() -> Message:
    content = RecordDict({
        "config": _model_config(),
        "model": _model_weights_arec(),
    })
    return Message(content, dst_node_id=0, message_type="query.fair1")


def _make_fair2_message(G: np.ndarray | None = None) -> Message:
    if G is None:
        G = np.zeros((D, D), dtype=np.float32)
    mu_s = np.zeros(D, dtype=np.float32)
    mu_f = np.zeros(D, dtype=np.float32)
    content = RecordDict({
        "config": _model_config(),
        "model": _model_weights_arec(),
        "fair2_data": ArrayRecord({
            "G": Array(G),
            "mu_s": Array(mu_s),
            "mu_f": Array(mu_f),
        }),
    })
    return Message(content, dst_node_id=0, message_type="query.fair2")


def _make_local_update_message(proximal_alpha: float = 1.0, num_local_epochs: int = 1) -> Message:
    content = RecordDict({
        "config": ConfigRecord({
            "round_num": 0,
            "step_size": 0.01,
            "proximal_alpha": proximal_alpha,
            "num_local_epochs": num_local_epochs,
            "input_dim": INPUT_DIM,
            "num_classes": NUM_CLASSES,
            "model_name": "logistic",
        }),
        "model": _model_weights_arec(),
    })
    return Message(content, dst_node_id=0, message_type="query.local_update")


# ---------------------------------------------------------------------------
# Tests for handle_fair1
# ---------------------------------------------------------------------------


class TestFair1Handler:
    @pytest.fixture(autouse=True)
    def patch_loader(self, monkeypatch):
        import kffl.app.client_app as ca
        monkeypatch.setattr(ca, "_load_partition", lambda ctx: _synthetic_loader())

    def _reply(self):
        from kffl.app.client_app import handle_fair1
        return handle_fair1(_make_fair1_message(), _StubContext())

    def test_returns_message(self):
        assert isinstance(self._reply(), Message)

    def test_reply_has_fair1_data_key(self):
        assert "fair1_data" in self._reply().content

    def test_reply_Mi_shape(self):
        arec: ArrayRecord = self._reply().content["fair1_data"]  # type: ignore[index]
        assert arec["Mi"].numpy().shape == (D, D)

    def test_reply_mu_s_shape(self):
        arec: ArrayRecord = self._reply().content["fair1_data"]  # type: ignore[index]
        assert arec["mu_s"].numpy().shape == (D,)

    def test_reply_mu_f_shape(self):
        arec: ArrayRecord = self._reply().content["fair1_data"]  # type: ignore[index]
        assert arec["mu_f"].numpy().shape == (D,)

    def test_reply_ni_equals_loader_size(self):
        arec: ArrayRecord = self._reply().content["fair1_data"]  # type: ignore[index]
        assert int(arec["ni"].numpy().item()) == N_SAMPLES

    def test_reply_Mi_dtype_is_float32(self):
        arec: ArrayRecord = self._reply().content["fair1_data"]  # type: ignore[index]
        assert arec["Mi"].numpy().dtype == np.float32

    def test_reply_Mi_is_finite(self):
        arec: ArrayRecord = self._reply().content["fair1_data"]  # type: ignore[index]
        assert np.isfinite(arec["Mi"].numpy()).all()


# ---------------------------------------------------------------------------
# Tests for handle_fair2
# ---------------------------------------------------------------------------


class TestFair2Handler:
    @pytest.fixture(autouse=True)
    def patch_loader(self, monkeypatch):
        import kffl.app.client_app as ca
        monkeypatch.setattr(ca, "_load_partition", lambda ctx: _synthetic_loader())

    def _reply(self):
        from kffl.app.client_app import handle_fair2
        return handle_fair2(_make_fair2_message(), _StubContext())

    def test_returns_message(self):
        assert isinstance(self._reply(), Message)

    def test_reply_has_gradient_key(self):
        assert "gradient" in self._reply().content

    def test_reply_gradient_has_param_keys(self):
        """Reply must have g_0, g_1, … matching number of model parameters."""
        arec: ArrayRecord = self._reply().content["gradient"]  # type: ignore[index]
        model = LogisticRegression(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
        n_params = len(list(model.parameters()))
        for i in range(n_params):
            assert f"g_{i}" in arec

    def test_reply_gradient_shapes_match_model(self):
        """Each gradient slice must match the corresponding parameter shape."""
        arec: ArrayRecord = self._reply().content["gradient"]  # type: ignore[index]
        model = LogisticRegression(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
        for i, p in enumerate(model.parameters()):
            assert arec[f"g_{i}"].numpy().shape == tuple(p.shape)

    def test_reply_gradients_are_finite(self):
        arec: ArrayRecord = self._reply().content["gradient"]  # type: ignore[index]
        model = LogisticRegression(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
        n_params = len(list(model.parameters()))
        for i in range(n_params):
            assert np.isfinite(arec[f"g_{i}"].numpy()).all()

    def test_reply_gradients_nonzero_for_nonzero_G(self):
        """With a non-zero G, gradients should generally be non-zero."""
        from kffl.app.client_app import handle_fair2

        msg = _make_fair2_message(G=np.eye(D, dtype=np.float32))
        reply = handle_fair2(msg, _StubContext())
        arec: ArrayRecord = reply.content["gradient"]  # type: ignore[index]
        total_grad_norm = sum(
            float(np.linalg.norm(arec[f"g_{i}"].numpy()))
            for i in range(len(list(LogisticRegression(INPUT_DIM).parameters())))
        )
        assert total_grad_norm > 0.0


# ---------------------------------------------------------------------------
# Tests for handle_local_update
# ---------------------------------------------------------------------------


class TestLocalUpdateHandler:
    @pytest.fixture(autouse=True)
    def patch_loader(self, monkeypatch):
        import kffl.app.client_app as ca
        monkeypatch.setattr(ca, "_load_partition", lambda ctx: _synthetic_loader())

    def _reply(self, **kwargs):
        from kffl.app.client_app import handle_local_update
        return handle_local_update(_make_local_update_message(**kwargs), _StubContext())

    def test_returns_message(self):
        assert isinstance(self._reply(), Message)

    def test_reply_has_model_key(self):
        assert "model" in self._reply().content

    def test_reply_has_weight_param_keys(self):
        """Reply must have w_0, w_1, … matching number of model parameters."""
        arec: ArrayRecord = self._reply().content["model"]  # type: ignore[index]
        model = LogisticRegression(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
        n_params = len(list(model.parameters()))
        for i in range(n_params):
            assert f"w_{i}" in arec

    def test_reply_weight_shapes_match_model(self):
        arec: ArrayRecord = self._reply().content["model"]  # type: ignore[index]
        model = LogisticRegression(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
        for i, p in enumerate(model.parameters()):
            assert arec[f"w_{i}"].numpy().shape == tuple(p.shape)

    def test_reply_weights_are_finite(self):
        arec: ArrayRecord = self._reply().content["model"]  # type: ignore[index]
        model = LogisticRegression(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
        n_params = len(list(model.parameters()))
        for i in range(n_params):
            assert np.isfinite(arec[f"w_{i}"].numpy()).all()

    def test_weights_change_after_training(self):
        """Local training should update the weights away from ω_{t+1/2}."""
        msg = _make_local_update_message(proximal_alpha=1.0, num_local_epochs=3)
        sent_arec: ArrayRecord = msg.content["model"]  # type: ignore[index]
        model = LogisticRegression(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
        n_params = len(list(model.parameters()))
        original = [sent_arec[f"w_{i}"].numpy() for i in range(n_params)]

        from kffl.app.client_app import handle_local_update
        reply = handle_local_update(msg, _StubContext())
        arec: ArrayRecord = reply.content["model"]  # type: ignore[index]

        any_changed = any(
            not np.allclose(arec[f"w_{i}"].numpy(), original[i])
            for i in range(n_params)
        )
        assert any_changed, "Weights should be updated by local training"
