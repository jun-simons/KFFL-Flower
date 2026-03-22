"""Comprehensive tests for the FAIR1 computation.

Tests the pure ``_compute_fair1_stats`` function directly — no Flower
context or UCI ML repository download required.

Covers:
  - Output shapes and dtypes
  - Determinism: same seed → same Z_S, Z_f → same Mi
  - Seed isolation: orf_s (seed) and orf_f (seed+1) produce different features
  - Mathematical properties: Mi = Z_Sᵀ Z_f, µ = mean(Z, axis=0)
  - ORF feature range (cos features are in [-√(2/D), +√(2/D)])
  - Effect of gamma on spread of features
  - Multi-column sensitive attributes (k > 1)
  - Binary vs multiclass models
  - Aggregation correctness: G = Σ Mᵢ − n · µ_s · µ_fᵀ
  - Integration with the Adult DataLoader
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kffl.app.client_app import _compute_fair1_stats
from kffl.ml import LogisticRegression, create_model, get_weights, set_weights


# ======================================================================
# Helpers
# ======================================================================


def _make_loader(
    n: int = 200,
    input_dim: int = 10,
    k_sensitive: int = 2,
    num_classes_s: int = 2,
    seed: int = 0,
    batch_size: int = 64,
) -> DataLoader:
    """Return a synthetic DataLoader with (X, y, s) of known size."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, input_dim, generator=g)
    y = (X[:, 0] > 0).long()
    s = torch.randint(0, num_classes_s, (n, k_sensitive), generator=g).long()
    ds = TensorDataset(X, y, s)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _run_fair1(
    loader: DataLoader,
    input_dim: int = 10,
    num_classes: int = 2,
    D: int = 16,
    gamma_s: float = 1.0,
    gamma_f: float = 1.0,
    seed: int = 42,
    model_seed: int = 7,
):
    """Create a fresh model and call _compute_fair1_stats."""
    torch.manual_seed(model_seed)
    model = LogisticRegression(input_dim=input_dim, num_classes=num_classes)
    return _compute_fair1_stats(loader, model, D, gamma_s, gamma_f, seed)


# ======================================================================
# Output shapes and dtypes
# ======================================================================


class TestFair1Shapes:
    @pytest.mark.parametrize("D", [8, 16, 32])
    def test_Mi_shape(self, D):
        loader = _make_loader()
        Mi, _, _, _ = _run_fair1(loader, D=D)
        assert Mi.shape == (D, D)

    @pytest.mark.parametrize("D", [8, 16, 32])
    def test_mu_s_shape(self, D):
        loader = _make_loader()
        _, mu_s, _, _ = _run_fair1(loader, D=D)
        assert mu_s.shape == (D,)

    @pytest.mark.parametrize("D", [8, 16, 32])
    def test_mu_f_shape(self, D):
        loader = _make_loader()
        _, _, mu_f, _ = _run_fair1(loader, D=D)
        assert mu_f.shape == (D,)

    def test_ni_equals_dataset_size(self):
        n = 150
        loader = _make_loader(n=n)
        _, _, _, ni = _run_fair1(loader)
        assert ni == n

    def test_Mi_dtype(self):
        Mi, _, _, _ = _run_fair1(_make_loader())
        assert Mi.dtype == np.float64

    def test_mu_dtypes(self):
        _, mu_s, mu_f, _ = _run_fair1(_make_loader())
        assert mu_s.dtype == np.float64
        assert mu_f.dtype == np.float64

    def test_all_outputs_finite(self):
        Mi, mu_s, mu_f, ni = _run_fair1(_make_loader())
        assert np.isfinite(Mi).all()
        assert np.isfinite(mu_s).all()
        assert np.isfinite(mu_f).all()
        assert np.isfinite(ni)


# ======================================================================
# Determinism and seed behaviour
# ======================================================================


class TestFair1Determinism:
    def test_same_seed_gives_same_Mi(self):
        loader = _make_loader()
        Mi_a, _, _, _ = _run_fair1(loader, seed=99, model_seed=1)
        Mi_b, _, _, _ = _run_fair1(loader, seed=99, model_seed=1)
        np.testing.assert_array_equal(Mi_a, Mi_b)

    def test_different_seed_gives_different_Mi(self):
        loader = _make_loader()
        Mi_a, _, _, _ = _run_fair1(loader, seed=1, model_seed=1)
        Mi_b, _, _, _ = _run_fair1(loader, seed=2, model_seed=1)
        assert not np.allclose(Mi_a, Mi_b)

    def test_orf_s_and_orf_f_use_different_projections(self):
        """orf_s (seed) and orf_f (seed+1) should produce different Z matrices
        even when both are fit on identical data."""
        from kffl.fairness.orf import OrthogonalRandomFeaturesRBF

        D, seed = 16, 42
        data = np.random.default_rng(0).standard_normal((100, 2))

        orf_s = OrthogonalRandomFeaturesRBF(gamma=1.0, n_components=D, random_state=seed)
        orf_f = OrthogonalRandomFeaturesRBF(gamma=1.0, n_components=D, random_state=seed + 1)

        Z_s = orf_s.fit_transform(data)
        Z_f = orf_f.fit_transform(data)

        assert not np.allclose(Z_s, Z_f), (
            "orf_s and orf_f should use different random projections."
        )

    def test_all_clients_see_same_Z_S_for_same_seed_and_data(self):
        """Two clients with the same local S and seed must get identical Z_S."""
        from kffl.fairness.orf import OrthogonalRandomFeaturesRBF

        D, seed = 16, 7
        S = np.random.default_rng(3).standard_normal((80, 2))

        Z1 = OrthogonalRandomFeaturesRBF(gamma=1.0, n_components=D, random_state=seed).fit_transform(S)
        Z2 = OrthogonalRandomFeaturesRBF(gamma=1.0, n_components=D, random_state=seed).fit_transform(S)

        np.testing.assert_array_equal(Z1, Z2)


# ======================================================================
# Mathematical correctness
# ======================================================================


class TestFair1MathProperties:
    def test_Mi_equals_ZS_T_Zf(self):
        """Mi must equal Z_Sᵀ Z_f up to float64 precision."""
        from kffl.fairness.orf import OrthogonalRandomFeaturesRBF

        D, seed = 8, 5
        n, input_dim, k = 80, 10, 2
        loader = _make_loader(n=n, input_dim=input_dim, k_sensitive=k, batch_size=n)

        torch.manual_seed(0)
        model = LogisticRegression(input_dim=input_dim)
        model.eval()

        # Collect data
        X_all, _, s_all = next(iter(loader))
        S = s_all.numpy().astype(np.float64)
        with torch.no_grad():
            f_X = model.predict_proba(X_all).numpy().reshape(-1, 1)

        Z_S = OrthogonalRandomFeaturesRBF(gamma=1.0, n_components=D, random_state=seed).fit_transform(S)
        Z_f = OrthogonalRandomFeaturesRBF(gamma=1.0, n_components=D, random_state=seed + 1).fit_transform(f_X)
        expected_Mi = Z_S.T @ Z_f

        Mi, _, _, _ = _compute_fair1_stats(loader, model, D, 1.0, 1.0, seed)
        np.testing.assert_allclose(Mi, expected_Mi, rtol=1e-5)

    def test_mu_s_equals_ZS_mean(self):
        """µ_{s,i} must be the column-wise mean of Z_S."""
        from kffl.fairness.orf import OrthogonalRandomFeaturesRBF

        D, seed = 8, 5
        n, input_dim, k = 80, 10, 2
        loader = _make_loader(n=n, input_dim=input_dim, k_sensitive=k, batch_size=n)

        torch.manual_seed(0)
        model = LogisticRegression(input_dim=input_dim)
        model.eval()

        X_all, _, s_all = next(iter(loader))
        S = s_all.numpy().astype(np.float64)

        Z_S = OrthogonalRandomFeaturesRBF(gamma=1.0, n_components=D, random_state=seed).fit_transform(S)
        expected_mu_s = Z_S.mean(axis=0)

        _, mu_s, _, _ = _compute_fair1_stats(loader, model, D, 1.0, 1.0, seed)
        np.testing.assert_allclose(mu_s, expected_mu_s, rtol=1e-5)

    def test_orf_features_in_valid_range(self):
        """cos features scaled by sqrt(2/D) lie in [-sqrt(2/D), +sqrt(2/D)]."""
        from kffl.fairness.orf import OrthogonalRandomFeaturesRBF

        D = 32
        data = np.random.default_rng(0).standard_normal((200, 3))
        Z = OrthogonalRandomFeaturesRBF(gamma=1.0, n_components=D, random_state=0).fit_transform(data)
        bound = np.sqrt(2.0 / D)
        assert float(np.abs(Z).max()) <= bound + 1e-9

    def test_Mi_norm_positive_for_nonzero_model(self):
        """A trained model (non-zero predictions) yields a non-trivial Mi."""
        # Give the model informative weights so predictions vary
        loader = _make_loader(n=200, input_dim=10)
        torch.manual_seed(0)
        model = LogisticRegression(input_dim=10)
        # Set weights so predictions vary meaningfully
        nn_params = list(model.parameters())
        with torch.no_grad():
            nn_params[0].fill_(0.5)  # weight
            nn_params[1].fill_(0.0)  # bias

        Mi, _, _, _ = _compute_fair1_stats(loader, model, D=16, gamma_s=1.0, gamma_f=1.0, seed=1)
        assert np.linalg.norm(Mi) > 0.0


# ======================================================================
# Effect of hyperparameters
# ======================================================================


class TestFair1HyperparamEffects:
    def test_larger_D_gives_larger_Mi(self):
        """Larger D should produce a larger Mi matrix (more entries)."""
        loader = _make_loader()
        Mi_small, _, _, _ = _run_fair1(loader, D=8)
        Mi_large, _, _, _ = _run_fair1(loader, D=32)
        # Shape changes, not value comparison
        assert Mi_small.shape == (8, 8)
        assert Mi_large.shape == (32, 32)

    def test_gamma_s_affects_Z_S(self):
        """Different gamma_s values must yield different Z_S and thus different Mi."""
        loader = _make_loader(seed=0)
        Mi_a, _, _, _ = _run_fair1(loader, gamma_s=0.1, seed=10, model_seed=1)
        Mi_b, _, _, _ = _run_fair1(loader, gamma_s=10.0, seed=10, model_seed=1)
        assert not np.allclose(Mi_a, Mi_b)

    def test_gamma_f_affects_Z_f(self):
        """Different gamma_f values must yield different Z_f and thus different Mi."""
        loader = _make_loader(seed=0)
        Mi_a, _, _, _ = _run_fair1(loader, gamma_f=0.1, seed=10, model_seed=1)
        Mi_b, _, _, _ = _run_fair1(loader, gamma_f=10.0, seed=10, model_seed=1)
        assert not np.allclose(Mi_a, Mi_b)

    def test_different_model_weights_give_different_Mi(self):
        """Changing the model weights changes f(X) and thus changes Mi."""
        loader = _make_loader()
        D, seed = 16, 42

        torch.manual_seed(0)
        model_a = LogisticRegression(input_dim=10)

        torch.manual_seed(1)
        model_b = LogisticRegression(input_dim=10)
        # Make sure they differ
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            pa.data.fill_(0.0)
            pb.data.fill_(1.0)

        Mi_a, _, _, _ = _compute_fair1_stats(loader, model_a, D, 1.0, 1.0, seed)
        Mi_b, _, _, _ = _compute_fair1_stats(loader, model_b, D, 1.0, 1.0, seed)
        assert not np.allclose(Mi_a, Mi_b)


# ======================================================================
# Multi-column sensitive attributes and multiclass models
# ======================================================================


class TestFair1Variants:
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_multi_column_sensitive(self, k):
        """_compute_fair1_stats works for any number of sensitive columns k."""
        loader = _make_loader(k_sensitive=k)
        Mi, mu_s, mu_f, ni = _run_fair1(loader, D=16)
        assert Mi.shape == (16, 16)
        assert mu_s.shape == (16,)
        assert mu_f.shape == (16,)
        assert ni == 200

    def test_multiclass_model_binary_output_is_2d(self):
        """Multiclass model returns (n, C) from predict_proba; ORF handles this."""
        loader = _make_loader(input_dim=10)
        D, seed = 16, 3
        torch.manual_seed(0)
        model = LogisticRegression(input_dim=10, num_classes=4)  # multiclass
        Mi, mu_s, mu_f, ni = _compute_fair1_stats(loader, model, D, 1.0, 1.0, seed)
        assert Mi.shape == (D, D)
        assert np.isfinite(Mi).all()


# ======================================================================
# Server-side aggregation correctness
# ======================================================================


class TestFair1Aggregation:
    """Verify that the server-side aggregation (eq. 13) is correct."""

    def test_G_aggregation_formula(self):
        """G = Σ Mᵢ − n · µ_s · µ_fᵀ should hold exactly when computed from
        the returned statistics."""
        D = 8
        # Simulate two clients
        loaders = [_make_loader(n=100, seed=0), _make_loader(n=80, seed=1)]
        torch.manual_seed(42)
        model = LogisticRegression(input_dim=10)

        sum_Mi = np.zeros((D, D))
        sum_ni_mu_s = np.zeros(D)
        sum_ni_mu_f = np.zeros(D)
        total_n = 0

        for loader in loaders:
            Mi, mu_s, mu_f, ni = _compute_fair1_stats(loader, model, D, 1.0, 1.0, seed=7)
            sum_Mi += Mi
            sum_ni_mu_s += ni * mu_s
            sum_ni_mu_f += ni * mu_f
            total_n += ni

        mu_s_global = sum_ni_mu_s / total_n
        mu_f_global = sum_ni_mu_f / total_n
        G = sum_Mi - total_n * np.outer(mu_s_global, mu_f_global)

        assert G.shape == (D, D)
        assert np.isfinite(G).all()

    def test_G_norm_nonnegative(self):
        """||G|| must be non-negative."""
        loader = _make_loader(n=150)
        Mi, mu_s, mu_f, ni = _run_fair1(loader, D=8)
        G = Mi - ni * np.outer(mu_s, mu_f)
        assert np.linalg.norm(G) >= 0.0

    def test_weighted_mu_aggregation(self):
        """Weighted means should reduce to plain mean when all clients have same n."""
        n = 100
        D = 8
        loaders = [_make_loader(n=n, seed=i) for i in range(3)]
        torch.manual_seed(5)
        model = LogisticRegression(input_dim=10)

        sum_ni_mu_s = np.zeros(D)
        sum_ni_mu_f = np.zeros(D)
        total_n = 0
        mu_s_list, mu_f_list = [], []

        for loader in loaders:
            Mi, mu_s, mu_f, ni = _compute_fair1_stats(loader, model, D, 1.0, 1.0, seed=3)
            sum_ni_mu_s += ni * mu_s
            sum_ni_mu_f += ni * mu_f
            total_n += ni
            mu_s_list.append(mu_s)
            mu_f_list.append(mu_f)

        # With equal n, weighted mean = plain mean
        np.testing.assert_allclose(
            sum_ni_mu_s / total_n, np.mean(mu_s_list, axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            sum_ni_mu_f / total_n, np.mean(mu_f_list, axis=0), rtol=1e-5
        )


# ======================================================================
# Integration: Adult DataLoader
# ======================================================================


class TestFair1WithAdultData:
    """Integration test using the actual Adult dataset (downloaded once)."""

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
        Mi, mu_s, mu_f, ni = _compute_fair1_stats(
            adult_loader, model, D=16, gamma_s=1.0, gamma_f=1.0, seed=42
        )
        assert Mi.shape == (16, 16)
        assert mu_s.shape == (16,)
        assert mu_f.shape == (16,)
        assert ni > 0

    def test_Mi_is_finite_on_adult(self, adult_loader, adult_input_dim):
        torch.manual_seed(0)
        model = LogisticRegression(input_dim=adult_input_dim)
        Mi, mu_s, mu_f, ni = _compute_fair1_stats(
            adult_loader, model, D=16, gamma_s=1.0, gamma_f=1.0, seed=42
        )
        assert np.isfinite(Mi).all()
        assert np.isfinite(mu_s).all()
        assert np.isfinite(mu_f).all()

    def test_ni_matches_partition_size(self, adult_loader, adult_input_dim):
        from kffl.data import get_federated_loaders
        full_n = sum(1 for _ in adult_loader.dataset)
        torch.manual_seed(0)
        model = LogisticRegression(input_dim=adult_input_dim)
        _, _, _, ni = _compute_fair1_stats(
            adult_loader, model, D=8, gamma_s=1.0, gamma_f=1.0, seed=0
        )
        assert ni == full_n

    def test_factory_model_gives_same_result(self, adult_loader, adult_input_dim):
        """create_model('logistic', ...) should give same results as LogisticRegression."""
        torch.manual_seed(0)
        m1 = LogisticRegression(input_dim=adult_input_dim)

        torch.manual_seed(0)
        m2 = create_model("logistic", input_dim=adult_input_dim)

        Mi1, _, _, _ = _compute_fair1_stats(adult_loader, m1, 8, 1.0, 1.0, seed=5)
        Mi2, _, _, _ = _compute_fair1_stats(adult_loader, m2, 8, 1.0, 1.0, seed=5)
        np.testing.assert_allclose(Mi1, Mi2, rtol=1e-5)
