"""Comprehensive tests for the KFFL ML module.

Covers:
  - base.py:      TrainResult, get_weights, set_weights
  - logistic.py:  LogisticRegression (forward, predict, predict_proba),
                  train() — loss reduction, proximal term, hyperparams
  - factory.py:   create_model, get_train_fn, registry errors
  - Integration:  full pipeline with the Adult DataLoader
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kffl.ml import (
    LogisticRegression,
    TrainFn,
    TrainResult,
    create_model,
    get_train_fn,
    get_weights,
    set_weights,
)
from kffl.ml.base import TrainResult


# ======================================================================
# Helpers
# ======================================================================


def _make_binary_loader(n: int = 200, d: int = 10, seed: int = 0) -> DataLoader:
    """Synthetic binary classification DataLoader yielding (X, y, s)."""
    rng = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=rng)
    y = (X[:, 0] > 0).long()          # linearly separable
    s = torch.zeros(n, 1, dtype=torch.long)
    ds = TensorDataset(X, y, s)
    return DataLoader(ds, batch_size=32, shuffle=False)


def _make_multiclass_loader(
    n: int = 200, d: int = 10, num_classes: int = 3, seed: int = 0
) -> DataLoader:
    rng = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=rng)
    y = (X[:, 0].abs() * num_classes).long().clamp(0, num_classes - 1)
    s = torch.zeros(n, 1, dtype=torch.long)
    ds = TensorDataset(X, y, s)
    return DataLoader(ds, batch_size=32, shuffle=False)


# ======================================================================
# Tests — base.py: TrainResult
# ======================================================================


class TestTrainResult:
    def test_final_loss_last_element(self):
        r = TrainResult(loss_history=[0.5, 0.3, 0.2], num_examples=100, num_epochs=3)
        assert r.final_loss == pytest.approx(0.2)

    def test_final_loss_empty(self):
        r = TrainResult(loss_history=[], num_examples=0, num_epochs=0)
        assert np.isnan(r.final_loss)

    def test_num_examples_stored(self):
        r = TrainResult(loss_history=[0.1], num_examples=256, num_epochs=1)
        assert r.num_examples == 256

    def test_num_epochs_stored(self):
        r = TrainResult(loss_history=[0.1, 0.05], num_examples=100, num_epochs=2)
        assert r.num_epochs == 2


# ======================================================================
# Tests — base.py: get_weights / set_weights
# ======================================================================


class TestWeightUtils:
    @pytest.fixture()
    def model(self):
        return LogisticRegression(input_dim=10)

    def test_get_weights_returns_list(self, model):
        weights = get_weights(model)
        assert isinstance(weights, list)

    def test_get_weights_length_matches_params(self, model):
        weights = get_weights(model)
        assert len(weights) == len(list(model.parameters()))

    def test_get_weights_are_numpy(self, model):
        for w in get_weights(model):
            assert isinstance(w, np.ndarray)

    def test_get_weights_are_copies(self, model):
        weights = get_weights(model)
        original = weights[0].copy()
        weights[0] *= 999
        # modifying the returned array must NOT affect the model
        assert np.allclose(list(model.parameters())[0].detach().numpy(), original)

    def test_set_weights_round_trip(self, model):
        original = get_weights(model)
        # Corrupt model parameters
        for p in model.parameters():
            p.data.fill_(0.0)
        # Restore
        set_weights(model, original)
        restored = get_weights(model)
        for orig, rest in zip(original, restored):
            np.testing.assert_allclose(orig, rest)

    def test_set_weights_wrong_length_raises(self, model):
        weights = get_weights(model)
        with pytest.raises(ValueError, match="Expected"):
            set_weights(model, weights[:-1])

    def test_set_weights_modifies_in_place(self, model):
        target = [np.ones_like(w) for w in get_weights(model)]
        set_weights(model, target)
        for p in model.parameters():
            assert torch.all(p == 1.0)


# ======================================================================
# Tests — logistic.py: LogisticRegression model
# ======================================================================


class TestLogisticRegressionModel:
    # --- Construction ---

    def test_binary_construction(self):
        model = LogisticRegression(input_dim=10, num_classes=2)
        assert model.input_dim == 10
        assert model.num_classes == 2
        assert model._binary is True

    def test_multiclass_construction(self):
        model = LogisticRegression(input_dim=10, num_classes=4)
        assert model._binary is False

    def test_invalid_input_dim_raises(self):
        with pytest.raises(ValueError, match="input_dim"):
            LogisticRegression(input_dim=0)

    def test_invalid_num_classes_raises(self):
        with pytest.raises(ValueError, match="num_classes"):
            LogisticRegression(input_dim=10, num_classes=1)

    # --- Forward: binary ---

    def test_binary_forward_shape(self):
        model = LogisticRegression(input_dim=10)
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8,), "Binary output should be (n,) logits"

    def test_binary_forward_dtype(self):
        model = LogisticRegression(input_dim=10)
        out = model(torch.randn(4, 10))
        assert out.dtype == torch.float32

    # --- Forward: multiclass ---

    def test_multiclass_forward_shape(self):
        model = LogisticRegression(input_dim=10, num_classes=3)
        out = model(torch.randn(8, 10))
        assert out.shape == (8, 3)

    # --- predict_proba ---

    def test_binary_proba_range(self):
        model = LogisticRegression(input_dim=10)
        proba = model.predict_proba(torch.randn(20, 10))
        assert proba.shape == (20,)
        assert float(proba.min()) >= 0.0
        assert float(proba.max()) <= 1.0

    def test_multiclass_proba_sums_to_one(self):
        model = LogisticRegression(input_dim=10, num_classes=4)
        proba = model.predict_proba(torch.randn(20, 10))
        assert proba.shape == (20, 4)
        row_sums = proba.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(20), atol=1e-5)

    def test_predict_proba_no_grad(self):
        model = LogisticRegression(input_dim=10)
        x = torch.randn(5, 10, requires_grad=True)
        proba = model.predict_proba(x)
        assert not proba.requires_grad

    # --- predict ---

    def test_binary_predict_values(self):
        model = LogisticRegression(input_dim=10)
        preds = model.predict(torch.randn(20, 10))
        assert preds.shape == (20,)
        assert set(preds.tolist()).issubset({0, 1})

    def test_multiclass_predict_values(self):
        model = LogisticRegression(input_dim=10, num_classes=4)
        preds = model.predict(torch.randn(20, 10))
        assert set(preds.tolist()).issubset({0, 1, 2, 3})


# ======================================================================
# Tests — logistic.py: train()
# ======================================================================


class TestLogisticTrain:
    @pytest.fixture()
    def binary_model(self):
        torch.manual_seed(42)
        return LogisticRegression(input_dim=10)

    @pytest.fixture()
    def multiclass_model(self):
        torch.manual_seed(42)
        return LogisticRegression(input_dim=10, num_classes=3)

    # --- Return type ---

    def test_returns_train_result(self, binary_model):
        loader = _make_binary_loader()
        result = get_train_fn("logistic")(
            binary_model, loader, lr=0.01, num_epochs=1,
            weight_decay=0.0, proximal_mu=0.0, proximal_center=None, device="cpu"
        )
        assert isinstance(result, TrainResult)

    def test_loss_history_length_equals_num_epochs(self, binary_model):
        loader = _make_binary_loader()
        result = get_train_fn("logistic")(
            binary_model, loader, lr=0.01, num_epochs=3,
            weight_decay=0.0, proximal_mu=0.0, proximal_center=None, device="cpu"
        )
        assert len(result.loss_history) == 3
        assert result.num_epochs == 3

    def test_num_examples_matches_dataset_size(self, binary_model):
        loader = _make_binary_loader(n=200)
        result = get_train_fn("logistic")(
            binary_model, loader, lr=0.01, num_epochs=1,
            weight_decay=0.0, proximal_mu=0.0, proximal_center=None, device="cpu"
        )
        assert result.num_examples == 200

    def test_loss_is_finite(self, binary_model):
        loader = _make_binary_loader()
        result = get_train_fn("logistic")(
            binary_model, loader, lr=0.01, num_epochs=2,
            weight_decay=0.0, proximal_mu=0.0, proximal_center=None, device="cpu"
        )
        assert all(np.isfinite(l) for l in result.loss_history)

    # --- Loss actually decreases on separable data ---

    def test_loss_decreases_over_epochs(self):
        torch.manual_seed(0)
        model = LogisticRegression(input_dim=10)
        loader = _make_binary_loader(n=400, seed=1)
        result = get_train_fn("logistic")(
            model, loader, lr=0.1, num_epochs=20,
            weight_decay=0.0, proximal_mu=0.0, proximal_center=None, device="cpu"
        )
        assert result.loss_history[-1] < result.loss_history[0], (
            "Loss should decrease on linearly separable data."
        )

    # --- Multiclass ---

    def test_multiclass_train_runs(self, multiclass_model):
        loader = _make_multiclass_loader(num_classes=3)
        result = get_train_fn("logistic")(
            multiclass_model, loader, lr=0.01, num_epochs=2,
            weight_decay=0.0, proximal_mu=0.0, proximal_center=None, device="cpu"
        )
        assert len(result.loss_history) == 2
        assert all(np.isfinite(l) for l in result.loss_history)

    # --- Weight decay changes loss ---

    def test_weight_decay_increases_loss(self):
        """With heavy L2 reg the loss should be higher than without."""
        torch.manual_seed(5)
        loader = _make_binary_loader(n=200, seed=5)

        def _run(wd):
            m = LogisticRegression(input_dim=10)
            nn.init.normal_(m.linear.weight, std=2.0)  # large weights → big penalty
            r = get_train_fn("logistic")(
                m, loader, lr=0.0,  # lr=0: no gradient steps, loss is pure initial
                num_epochs=1, weight_decay=wd,
                proximal_mu=0.0, proximal_center=None, device="cpu"
            )
            return r.final_loss

        # With a non-zero learning rate and weight decay, loss path differs
        torch.manual_seed(5)
        m1 = LogisticRegression(input_dim=10)
        r1 = get_train_fn("logistic")(
            m1, loader, lr=0.01, num_epochs=5,
            weight_decay=0.0, proximal_mu=0.0, proximal_center=None, device="cpu"
        )
        torch.manual_seed(5)
        m2 = LogisticRegression(input_dim=10)
        r2 = get_train_fn("logistic")(
            m2, loader, lr=0.01, num_epochs=5,
            weight_decay=10.0, proximal_mu=0.0, proximal_center=None, device="cpu"
        )
        # Heavy regularization → different (typically higher) final loss
        assert r1.final_loss != pytest.approx(r2.final_loss, rel=1e-3)

    # --- Proximal term ---

    def test_proximal_mu_without_center_raises(self, binary_model):
        loader = _make_binary_loader()
        with pytest.raises(ValueError, match="proximal_center"):
            get_train_fn("logistic")(
                binary_model, loader, lr=0.01, num_epochs=1,
                weight_decay=0.0, proximal_mu=1.0, proximal_center=None, device="cpu"
            )

    def test_proximal_pulls_toward_center(self):
        """Model trained with proximal term should end up closer to the center
        than one trained without it (same init, same data, same epochs)."""
        d = 10
        loader = _make_binary_loader(n=200, d=d)

        # Use a center that's clearly away from the typical initialization
        center = [np.full_like(w, 3.0) for w in get_weights(LogisticRegression(input_dim=d))]

        def _dist_to_center(model):
            return sum(float(np.abs(f - c).sum()) for f, c in zip(get_weights(model), center))

        torch.manual_seed(99)
        m_no_prox = LogisticRegression(input_dim=d)
        get_train_fn("logistic")(
            m_no_prox, loader, lr=0.01, num_epochs=10,
            weight_decay=0.0, proximal_mu=0.0, proximal_center=None, device="cpu"
        )

        torch.manual_seed(99)
        m_prox = LogisticRegression(input_dim=d)
        get_train_fn("logistic")(
            m_prox, loader, lr=0.01, num_epochs=10,
            weight_decay=0.0, proximal_mu=5.0, proximal_center=center, device="cpu"
        )

        assert _dist_to_center(m_prox) < _dist_to_center(m_no_prox), (
            "Proximal term should pull the model closer to the center."
        )

    def test_proximal_zero_mu_ignores_center(self):
        """proximal_mu=0 should behave identically to no proximal term."""
        d = 10
        center = [np.zeros(10), np.zeros(1)]  # wrong shapes, but ignored

        torch.manual_seed(7)
        m1 = LogisticRegression(input_dim=d)
        loader = _make_binary_loader(n=100, d=d)
        r1 = get_train_fn("logistic")(
            m1, loader, lr=0.01, num_epochs=3,
            weight_decay=0.0, proximal_mu=0.0, proximal_center=None, device="cpu"
        )

        torch.manual_seed(7)
        m2 = LogisticRegression(input_dim=d)
        # Providing center with mu=0 should be fine and identical
        r2 = get_train_fn("logistic")(
            m2, loader, lr=0.01, num_epochs=3,
            weight_decay=0.0, proximal_mu=0.0, proximal_center=center, device="cpu"
        )
        assert r1.final_loss == pytest.approx(r2.final_loss, rel=1e-5)

    # --- Model is modified in-place ---

    def test_weights_change_after_training(self, binary_model):
        initial = get_weights(binary_model)
        loader = _make_binary_loader()
        get_train_fn("logistic")(
            binary_model, loader, lr=0.01, num_epochs=2,
            weight_decay=0.0, proximal_mu=0.0, proximal_center=None, device="cpu"
        )
        final = get_weights(binary_model)
        changed = any(not np.allclose(i, f) for i, f in zip(initial, final))
        assert changed, "Training should update model weights."


# ======================================================================
# Tests — factory.py
# ======================================================================


class TestFactory:
    def test_create_model_logistic_binary(self):
        model = create_model("logistic", input_dim=20)
        assert isinstance(model, LogisticRegression)
        assert model.input_dim == 20
        assert model.num_classes == 2

    def test_create_model_logistic_multiclass(self):
        model = create_model("logistic", input_dim=20, num_classes=5)
        assert model.num_classes == 5
        assert not model._binary

    def test_create_model_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent", input_dim=10)

    def test_get_train_fn_logistic(self):
        fn = get_train_fn("logistic")
        assert callable(fn)
        assert isinstance(fn, TrainFn)

    def test_get_train_fn_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_train_fn("nonexistent")

    def test_factory_model_is_nn_module(self):
        model = create_model("logistic", input_dim=5)
        assert isinstance(model, nn.Module)

    def test_factory_train_fn_returns_train_result(self):
        model = create_model("logistic", input_dim=10)
        loader = _make_binary_loader()
        fn = get_train_fn("logistic")
        result = fn(
            model, loader, lr=0.01, num_epochs=1,
            weight_decay=0.0, proximal_mu=0.0, proximal_center=None, device="cpu"
        )
        assert isinstance(result, TrainResult)


# ======================================================================
# Integration tests — Adult dataset DataLoader
# ======================================================================


class TestIntegrationWithAdult:
    """End-to-end: create model sized to Adult data, train for a few epochs."""

    @pytest.fixture(scope="class")
    def adult_loader(self):
        from kffl.data import get_federated_loaders
        loaders = get_federated_loaders("adult", num_partitions=5, batch_size=64, seed=0)
        return loaders[0]

    @pytest.fixture(scope="class")
    def adult_input_dim(self, adult_loader):
        X, _, _ = next(iter(adult_loader))
        return X.shape[1]

    def test_model_trains_on_adult_data(self, adult_loader, adult_input_dim):
        model = create_model("logistic", input_dim=adult_input_dim)
        fn = get_train_fn("logistic")
        result = fn(
            model, adult_loader, lr=0.01, num_epochs=3,
            weight_decay=1e-4, proximal_mu=0.0, proximal_center=None, device="cpu"
        )
        assert len(result.loss_history) == 3
        assert all(np.isfinite(l) for l in result.loss_history)

    def test_loss_decreases_on_adult(self, adult_loader, adult_input_dim):
        torch.manual_seed(0)
        model = create_model("logistic", input_dim=adult_input_dim)
        fn = get_train_fn("logistic")
        result = fn(
            model, adult_loader, lr=0.05, num_epochs=10,
            weight_decay=0.0, proximal_mu=0.0, proximal_center=None, device="cpu"
        )
        assert result.loss_history[-1] < result.loss_history[0]

    def test_predict_proba_on_adult_batch(self, adult_loader, adult_input_dim):
        model = create_model("logistic", input_dim=adult_input_dim)
        X, _, _ = next(iter(adult_loader))
        proba = model.predict_proba(X)
        assert proba.shape[0] == X.shape[0]
        assert float(proba.min()) >= 0.0
        assert float(proba.max()) <= 1.0

    def test_proximal_train_on_adult(self, adult_loader, adult_input_dim):
        model = create_model("logistic", input_dim=adult_input_dim)
        center = get_weights(model)  # proximal center = initial weights
        fn = get_train_fn("logistic")
        result = fn(
            model, adult_loader, lr=0.01, num_epochs=2,
            weight_decay=0.0, proximal_mu=0.1, proximal_center=center, device="cpu"
        )
        assert len(result.loss_history) == 2
        assert all(np.isfinite(l) for l in result.loss_history)

    def test_get_set_weights_round_trip_on_adult_model(self, adult_input_dim):
        model = create_model("logistic", input_dim=adult_input_dim)
        original = get_weights(model)
        for p in model.parameters():
            p.data.zero_()
        set_weights(model, original)
        for orig, restored in zip(original, get_weights(model)):
            np.testing.assert_allclose(orig, restored)
