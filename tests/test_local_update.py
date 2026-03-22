"""Comprehensive tests for the KFFL local update step.

The local update solves (eq. 21):

    ωᵢ_{t+1} = argmin_ω [ fᵢ(ω) + (μ/2) · ‖ω − ω_{t+1/2}‖² ]

via proximal SGD.  The pure ``train()`` function is tested here directly
(no Flower context / UCI download required for most tests).

Covers:
  - Weights change after local training
  - Returned weights have correct shapes and dtypes
  - Loss decreases across epochs
  - Proximal term pulls weights toward the centre (ω_{t+1/2})
  - Strong proximal penalty keeps weights very close to centre
  - μ = 0 recovers standard ERM training (weights still update)
  - Determinism: same seed + data → same result
  - Multi-epoch training converges further than single-epoch
  - Server averaging of multiple client updates
  - Integration with Adult DataLoader
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kffl.ml import LogisticRegression, get_weights, set_weights
from kffl.ml.logistic import train


# ======================================================================
# Helpers
# ======================================================================

INPUT_DIM = 10
NUM_CLASSES = 2
N_SAMPLES = 200


def _make_loader(
    n: int = N_SAMPLES,
    input_dim: int = INPUT_DIM,
    seed: int = 0,
    batch_size: int = 64,
) -> DataLoader:
    """Synthetic DataLoader yielding (X, y, s) triples."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, input_dim, generator=g)
    y = (X[:, 0] > 0).long()
    # 2-column sensitive attribute (ignored during ERM, accepted by train())
    s = torch.zeros(n, 2, dtype=torch.long)
    return DataLoader(TensorDataset(X, y, s), batch_size=batch_size, shuffle=False)


def _fresh_model(input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES, seed: int = 42) -> LogisticRegression:
    torch.manual_seed(seed)
    return LogisticRegression(input_dim=input_dim, num_classes=num_classes)


def _run_train(
    model: LogisticRegression,
    loader: DataLoader,
    *,
    lr: float = 0.05,
    num_epochs: int = 1,
    proximal_mu: float = 0.0,
    proximal_center: List[np.ndarray] | None = None,
):
    """Thin wrapper around train() with sensible defaults."""
    return train(
        model,
        loader,
        lr=lr,
        num_epochs=num_epochs,
        proximal_mu=proximal_mu,
        proximal_center=proximal_center,
    )


# ======================================================================
# Basic correctness: weights change and shapes are preserved
# ======================================================================


class TestLocalUpdateBasics:
    def test_weights_change_after_training(self):
        """Weights must differ from ω_{t+1/2} after at least one SGD step."""
        loader = _make_loader()
        model = _fresh_model()
        w_before = get_weights(model)
        _run_train(model, loader, num_epochs=1)
        w_after = get_weights(model)
        any_changed = any(not np.allclose(a, b) for a, b in zip(w_after, w_before))
        assert any_changed

    def test_weight_shapes_preserved(self):
        """Parameter shapes must be identical before and after training."""
        loader = _make_loader()
        model = _fresh_model()
        shapes_before = [w.shape for w in get_weights(model)]
        _run_train(model, loader, num_epochs=2)
        shapes_after = [w.shape for w in get_weights(model)]
        assert shapes_before == shapes_after

    def test_weights_are_finite(self):
        """All returned parameters must be finite after training."""
        loader = _make_loader()
        model = _fresh_model()
        _run_train(model, loader, num_epochs=3)
        for w in get_weights(model):
            assert np.isfinite(w).all()

    def test_weight_dtype_float32(self):
        loader = _make_loader()
        model = _fresh_model()
        _run_train(model, loader)
        for w in get_weights(model):
            assert w.dtype == np.float32

    def test_returns_train_result(self):
        from kffl.ml import TrainResult
        loader = _make_loader()
        model = _fresh_model()
        result = _run_train(model, loader)
        assert isinstance(result, TrainResult)

    def test_loss_history_length(self):
        loader = _make_loader()
        model = _fresh_model()
        num_epochs = 4
        result = _run_train(model, loader, num_epochs=num_epochs)
        assert len(result.loss_history) == num_epochs

    def test_loss_is_finite(self):
        loader = _make_loader()
        model = _fresh_model()
        result = _run_train(model, loader, num_epochs=3)
        for loss in result.loss_history:
            assert np.isfinite(loss)

    def test_num_examples_matches_dataset(self):
        n = 150
        loader = _make_loader(n=n)
        model = _fresh_model()
        result = _run_train(model, loader)
        assert result.num_examples == n


# ======================================================================
# Loss convergence
# ======================================================================


class TestLocalUpdateConvergence:
    def test_loss_decreases_over_epochs(self):
        """With enough data and a reasonable lr, loss should be lower at the end."""
        loader = _make_loader(n=500, seed=1)
        model = _fresh_model(seed=0)
        result = _run_train(model, loader, lr=0.1, num_epochs=10)
        assert result.loss_history[-1] < result.loss_history[0], (
            f"Expected loss to decrease, got {result.loss_history}"
        )

    def test_more_epochs_give_lower_final_loss(self):
        """Training for more epochs should reach a lower loss from the same start."""
        loader = _make_loader(n=300, seed=2)

        torch.manual_seed(5)
        model_short = LogisticRegression(input_dim=INPUT_DIM)
        result_short = _run_train(model_short, loader, lr=0.1, num_epochs=1)

        torch.manual_seed(5)
        model_long = LogisticRegression(input_dim=INPUT_DIM)
        result_long = _run_train(model_long, loader, lr=0.1, num_epochs=15)

        assert result_long.final_loss <= result_short.final_loss

    def test_multiclass_loss_decreases(self):
        """Loss should decrease for a multiclass model too."""
        loader = _make_loader(n=400, seed=3)
        torch.manual_seed(0)
        model = LogisticRegression(input_dim=INPUT_DIM, num_classes=4)
        result = _run_train(model, loader, lr=0.1, num_epochs=10)
        assert result.loss_history[-1] < result.loss_history[0]


# ======================================================================
# Proximal penalty
# ======================================================================


class TestLocalUpdateProximal:
    def test_proximal_mu_without_center_raises(self):
        loader = _make_loader()
        model = _fresh_model()
        with pytest.raises(ValueError, match="proximal_center"):
            train(model, loader, proximal_mu=1.0)

    def test_alpha_to_mu_conversion(self):
        """The handler converts proximal_alpha → proximal_mu = 1/α.
        Larger α means weaker penalty (smaller mu), so weights drift further."""
        loader = _make_loader(n=300, seed=5)

        def _drift(proximal_alpha: float) -> float:
            # Reproduce the handler's conversion: proximal_mu = 1 / proximal_alpha
            proximal_mu = 1.0 / proximal_alpha
            torch.manual_seed(0)
            model = LogisticRegression(input_dim=INPUT_DIM)
            center = get_weights(model)
            train(model, loader, lr=0.05, num_epochs=3,
                  proximal_mu=proximal_mu, proximal_center=center)
            return float(np.sqrt(sum(
                np.sum((a - b) ** 2) for a, b in zip(get_weights(model), center)
            )))

        # Small α → large μ (= 1/α) → strong penalty → less drift
        # Large α → small μ (= 1/α) → weak penalty → more drift
        drift_small_alpha = _drift(proximal_alpha=0.1)   # μ = 10
        drift_large_alpha = _drift(proximal_alpha=10.0)  # μ = 0.1
        assert drift_small_alpha < drift_large_alpha, (
            f"Small α should give less drift: {drift_small_alpha:.4f} vs {drift_large_alpha:.4f}"
        )

    def test_strong_proximal_keeps_weights_near_center(self):
        """Very large μ pins the model close to the proximal centre.

        For stability, lr must satisfy lr · μ ≪ 1.  Here μ=500, lr=1e-3
        gives μ·lr = 0.5, well within the convergent regime.
        """
        loader = _make_loader(n=300, seed=7)
        model = _fresh_model()
        center = get_weights(model)  # snapshot before training

        train(
            model, loader,
            lr=1e-3, num_epochs=10,
            proximal_mu=500.0,
            proximal_center=center,
        )

        w_after = get_weights(model)
        total_drift = float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(w_after, center))))
        assert total_drift < 5e-2, f"Weights drifted too far: {total_drift}"

    def test_proximal_term_reduces_drift_vs_no_penalty(self):
        """Weights should stay closer to centre with proximal term than without."""
        loader = _make_loader(n=300, seed=8)

        # Without proximal term
        torch.manual_seed(10)
        model_free = LogisticRegression(input_dim=INPUT_DIM)
        center = get_weights(model_free)
        train(model_free, loader, lr=0.1, num_epochs=5)
        drift_free = float(np.sqrt(sum(
            np.sum((a - b) ** 2) for a, b in zip(get_weights(model_free), center)
        )))

        # With proximal term (same init, same center)
        torch.manual_seed(10)
        model_prox = LogisticRegression(input_dim=INPUT_DIM)
        center_prox = get_weights(model_prox)
        train(model_prox, loader, lr=0.1, num_epochs=5,
              proximal_mu=5.0, proximal_center=center_prox)
        drift_prox = float(np.sqrt(sum(
            np.sum((a - b) ** 2) for a, b in zip(get_weights(model_prox), center_prox)
        )))

        assert drift_prox < drift_free, (
            f"Proximal drift {drift_prox:.4f} should be less than free drift {drift_free:.4f}"
        )

    def test_zero_proximal_mu_trains_normally(self):
        """μ = 0 should give the same result as calling train without proximal args."""
        loader = _make_loader(n=200, seed=9)

        torch.manual_seed(3)
        model_a = LogisticRegression(input_dim=INPUT_DIM)
        result_a = train(model_a, loader, lr=0.05, num_epochs=3)

        torch.manual_seed(3)
        model_b = LogisticRegression(input_dim=INPUT_DIM)
        # proximal_mu=0 with a center provided — centre should have no effect
        center = get_weights(model_b)
        result_b = train(model_b, loader, lr=0.05, num_epochs=3,
                         proximal_mu=0.0, proximal_center=center)

        for wa, wb in zip(get_weights(model_a), get_weights(model_b)):
            np.testing.assert_allclose(wa, wb, rtol=1e-5)

    def test_proximal_centre_is_omega_half(self):
        """The proximal centre is ω_{t+1/2}: setting center = initial weights and
        running one epoch should give a result pulled toward that start point."""
        loader = _make_loader(n=200, seed=11)
        torch.manual_seed(0)
        model = LogisticRegression(input_dim=INPUT_DIM)
        omega_half = get_weights(model)

        # Without proximal — weights can move freely
        torch.manual_seed(0)
        model_free = LogisticRegression(input_dim=INPUT_DIM)
        train(model_free, loader, lr=0.2, num_epochs=3)
        dist_free = float(np.sqrt(sum(
            np.sum((a - b) ** 2)
            for a, b in zip(get_weights(model_free), omega_half)
        )))

        # With proximal anchored at omega_half
        torch.manual_seed(0)
        model_prox = LogisticRegression(input_dim=INPUT_DIM)
        train(model_prox, loader, lr=0.2, num_epochs=3,
              proximal_mu=10.0, proximal_center=omega_half)
        dist_prox = float(np.sqrt(sum(
            np.sum((a - b) ** 2)
            for a, b in zip(get_weights(model_prox), omega_half)
        )))

        assert dist_prox < dist_free


# ======================================================================
# Determinism
# ======================================================================


class TestLocalUpdateDeterminism:
    def test_same_inputs_give_identical_weights(self):
        """Two calls with the same model state and data must produce the same result."""
        loader = _make_loader(seed=42)

        torch.manual_seed(7)
        model_a = LogisticRegression(input_dim=INPUT_DIM)
        center = get_weights(model_a)
        train(model_a, loader, lr=0.05, num_epochs=3,
              proximal_mu=0.5, proximal_center=center)

        torch.manual_seed(7)
        model_b = LogisticRegression(input_dim=INPUT_DIM)
        center_b = get_weights(model_b)
        train(model_b, loader, lr=0.05, num_epochs=3,
              proximal_mu=0.5, proximal_center=center_b)

        for wa, wb in zip(get_weights(model_a), get_weights(model_b)):
            np.testing.assert_array_equal(wa, wb)

    def test_different_data_gives_different_weights(self):
        loader_a = _make_loader(seed=0)
        loader_b = _make_loader(seed=999)

        torch.manual_seed(1)
        model_a = LogisticRegression(input_dim=INPUT_DIM)
        train(model_a, loader_a, lr=0.1, num_epochs=3)

        torch.manual_seed(1)
        model_b = LogisticRegression(input_dim=INPUT_DIM)
        train(model_b, loader_b, lr=0.1, num_epochs=3)

        any_differ = any(
            not np.allclose(a, b)
            for a, b in zip(get_weights(model_a), get_weights(model_b))
        )
        assert any_differ


# ======================================================================
# Server-side weight averaging (FedAvg component)
# ======================================================================


class TestLocalUpdateServerAveraging:
    """The server averages per-client ωᵢ_{t+1} to obtain ω_{t+1}."""

    def _client_update(self, loader: DataLoader, center: List[np.ndarray],
                       seed: int, proximal_mu: float = 0.1) -> List[np.ndarray]:
        torch.manual_seed(seed)
        model = LogisticRegression(input_dim=INPUT_DIM)
        set_weights(model, center)
        train(model, loader, lr=0.05, num_epochs=2,
              proximal_mu=proximal_mu, proximal_center=center)
        return get_weights(model)

    def test_average_of_two_identical_clients_is_itself(self):
        """Averaging two clients with the same data and model is idempotent."""
        loader = _make_loader(seed=0)
        torch.manual_seed(99)
        model = LogisticRegression(input_dim=INPUT_DIM)
        center = get_weights(model)

        wa = self._client_update(loader, center, seed=99)
        wb = self._client_update(loader, center, seed=99)

        avg = [np.mean([a, b], axis=0) for a, b in zip(wa, wb)]
        for a, b in zip(avg, wa):
            np.testing.assert_allclose(a, b, rtol=1e-5)

    def test_average_lies_between_client_weights(self):
        """Each averaged parameter should lie between the two client extremes."""
        loaders = [_make_loader(seed=0), _make_loader(seed=1)]
        torch.manual_seed(0)
        model = LogisticRegression(input_dim=INPUT_DIM)
        center = get_weights(model)

        client_weights = [self._client_update(l, center, seed=i) for i, l in enumerate(loaders)]
        avg = [
            np.mean([cw[j] for cw in client_weights], axis=0)
            for j in range(len(center))
        ]

        for j, (a, c1, c2) in enumerate(zip(avg, client_weights[0], client_weights[1])):
            lo = np.minimum(c1, c2)
            hi = np.maximum(c1, c2)
            assert np.all(a >= lo - 1e-6) and np.all(a <= hi + 1e-6), (
                f"Param {j}: averaged weight outside [min, max] of client weights"
            )

    def test_server_model_updated_with_avg(self):
        """set_weights(model, avg) correctly installs the averaged parameters."""
        loaders = [_make_loader(seed=i) for i in range(3)]
        torch.manual_seed(0)
        server_model = LogisticRegression(input_dim=INPUT_DIM)
        center = get_weights(server_model)

        client_weights = [self._client_update(l, center, seed=i) for i, l in enumerate(loaders)]
        n = len(client_weights)
        avg = [
            np.mean([cw[j] for cw in client_weights], axis=0)
            for j in range(len(center))
        ]

        set_weights(server_model, avg)
        for w_model, w_avg in zip(get_weights(server_model), avg):
            np.testing.assert_allclose(w_model, w_avg, rtol=1e-6)

    def test_averaging_more_clients_is_stable(self):
        """Averaging 5 client updates should produce a finite, valid model."""
        loaders = [_make_loader(seed=i, n=100) for i in range(5)]
        torch.manual_seed(0)
        model = LogisticRegression(input_dim=INPUT_DIM)
        center = get_weights(model)

        all_weights = [self._client_update(l, center, seed=i) for i, l in enumerate(loaders)]
        avg = [
            np.mean([cw[j] for cw in all_weights], axis=0)
            for j in range(len(center))
        ]
        for w in avg:
            assert np.isfinite(w).all()


# ======================================================================
# Integration: Adult DataLoader
# ======================================================================


class TestLocalUpdateWithAdultData:
    """Integration test using the real Adult dataset."""

    @pytest.fixture(scope="class")
    def adult_loader(self):
        from kffl.data import get_federated_loaders
        return get_federated_loaders("adult", num_partitions=5, batch_size=64)[0]

    @pytest.fixture(scope="class")
    def adult_input_dim(self, adult_loader):
        X, _, _ = next(iter(adult_loader))
        return X.shape[1]

    def test_runs_without_error(self, adult_loader, adult_input_dim):
        torch.manual_seed(0)
        model = LogisticRegression(input_dim=adult_input_dim)
        result = train(model, adult_loader, lr=0.01, num_epochs=1)
        assert result.num_examples > 0

    def test_weights_finite_on_adult(self, adult_loader, adult_input_dim):
        torch.manual_seed(0)
        model = LogisticRegression(input_dim=adult_input_dim)
        train(model, adult_loader, lr=0.01, num_epochs=2)
        for w in get_weights(model):
            assert np.isfinite(w).all()

    def test_loss_decreases_on_adult(self, adult_loader, adult_input_dim):
        torch.manual_seed(0)
        model = LogisticRegression(input_dim=adult_input_dim)
        result = train(model, adult_loader, lr=0.05, num_epochs=5)
        assert result.loss_history[-1] < result.loss_history[0]

    def test_proximal_train_on_adult(self, adult_loader, adult_input_dim):
        """Proximal training on Adult data should run and produce finite weights."""
        torch.manual_seed(0)
        model = LogisticRegression(input_dim=adult_input_dim)
        center = get_weights(model)
        result = train(
            model, adult_loader,
            lr=0.01, num_epochs=2,
            proximal_mu=0.1, proximal_center=center,
        )
        assert result.num_examples > 0
        for w in get_weights(model):
            assert np.isfinite(w).all()

    def test_proximal_keeps_weights_closer_to_center_on_adult(self, adult_loader, adult_input_dim):
        """Proximal penalty should produce less drift than free training on Adult."""
        # Free training
        torch.manual_seed(0)
        model_free = LogisticRegression(input_dim=adult_input_dim)
        center = get_weights(model_free)
        train(model_free, adult_loader, lr=0.1, num_epochs=5)
        drift_free = float(np.sqrt(sum(
            np.sum((a - b) ** 2) for a, b in zip(get_weights(model_free), center)
        )))

        # Proximal training
        torch.manual_seed(0)
        model_prox = LogisticRegression(input_dim=adult_input_dim)
        center_p = get_weights(model_prox)
        train(model_prox, adult_loader, lr=0.1, num_epochs=5,
              proximal_mu=5.0, proximal_center=center_p)
        drift_prox = float(np.sqrt(sum(
            np.sum((a - b) ** 2) for a, b in zip(get_weights(model_prox), center_p)
        )))

        assert drift_prox < drift_free
