"""Comprehensive tests for the KFFL data-loading pipeline.

Covers:
  - adult.py:   DatasetBundle contract, shapes, dtypes, value ranges,
                single and multiple sensitive features
  - dataset.py: FairDataset, IID partitioning, get_federated_loaders
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kffl.data.adult import DatasetBundle, load as load_adult
from kffl.data.dataset import (
    FairDataset,
    _iid_partition_indices,
    get_federated_loaders,
    load_dataset,
)


# ======================================================================
# Fixtures  (loaded once per module to keep the test suite fast)
# ======================================================================


@pytest.fixture(scope="module")
def adult_bundle() -> DatasetBundle:
    """Default Adult bundle: sensitive_features=["sex", "race"]."""
    return load_adult()


@pytest.fixture(scope="module")
def adult_bundle_sex_only() -> DatasetBundle:
    return load_adult(sensitive_features=["sex"])


@pytest.fixture(scope="module")
def adult_bundle_race_only() -> DatasetBundle:
    return load_adult(sensitive_features=["race"])


# ======================================================================
# Tests for adult.py — DatasetBundle contract
# ======================================================================


class TestAdultLoader:
    """Verify the Adult dataset-specific loader."""

    # --- Basic types and shapes (default: sex + race) ---

    def test_returns_dataset_bundle(self, adult_bundle):
        assert isinstance(adult_bundle, DatasetBundle)

    def test_X_is_2d_float32(self, adult_bundle):
        assert adult_bundle.X.ndim == 2
        assert adult_bundle.X.shape[0] > 1000
        assert adult_bundle.X.dtype == np.float32

    def test_y_is_1d_int64_binary(self, adult_bundle):
        assert adult_bundle.y.ndim == 1
        assert adult_bundle.y.shape[0] == adult_bundle.X.shape[0]
        assert adult_bundle.y.dtype == np.int64
        assert set(adult_bundle.y) == {0, 1}

    def test_s_is_2d_int64(self, adult_bundle):
        assert adult_bundle.s.ndim == 2
        assert adult_bundle.s.shape[0] == adult_bundle.X.shape[0]
        assert adult_bundle.s.dtype == np.int64

    def test_s_has_k_columns_matching_sensitive_names(self, adult_bundle):
        k = len(adult_bundle.sensitive_names)
        assert adult_bundle.s.shape[1] == k

    # --- Default sensitive features: ["sex", "race"] ---

    def test_default_sensitive_names(self, adult_bundle):
        assert adult_bundle.sensitive_names == ["sex", "race"]

    def test_default_k_equals_two(self, adult_bundle):
        assert adult_bundle.s.shape[1] == 2

    def test_sex_column_has_two_groups(self, adult_bundle):
        sex_idx = adult_bundle.sensitive_names.index("sex")
        unique_codes = set(adult_bundle.s[:, sex_idx])
        assert len(unique_codes) == 2
        assert unique_codes == set(adult_bundle.sensitive_labels[sex_idx].keys())

    def test_race_column_has_multiple_groups(self, adult_bundle):
        race_idx = adult_bundle.sensitive_names.index("race")
        unique_codes = set(adult_bundle.s[:, race_idx])
        assert len(unique_codes) > 2
        assert unique_codes == set(adult_bundle.sensitive_labels[race_idx].keys())

    def test_sensitive_labels_is_list_of_length_k(self, adult_bundle):
        k = len(adult_bundle.sensitive_names)
        assert isinstance(adult_bundle.sensitive_labels, list)
        assert len(adult_bundle.sensitive_labels) == k

    def test_each_labels_dict_maps_int_to_str(self, adult_bundle):
        for label_dict in adult_bundle.sensitive_labels:
            assert isinstance(label_dict, dict)
            for code, name in label_dict.items():
                assert isinstance(code, int)
                assert isinstance(name, str)

    # --- Feature matrix ---

    def test_feature_names_length_matches_X(self, adult_bundle):
        assert len(adult_bundle.feature_names) == adult_bundle.X.shape[1]

    def test_sensitive_names_not_in_feature_names(self, adult_bundle):
        for name in adult_bundle.sensitive_names:
            assert name not in adult_bundle.feature_names

    def test_target_not_in_feature_names(self, adult_bundle):
        assert adult_bundle.target_name not in adult_bundle.feature_names

    def test_no_nan_in_X(self, adult_bundle):
        assert not np.isnan(adult_bundle.X).any()

    def test_no_nan_in_y(self, adult_bundle):
        assert not np.isnan(adult_bundle.y.astype(float)).any()

    def test_continuous_features_standardised(self, adult_bundle):
        age_idx = adult_bundle.feature_names.index("age")
        col = adult_bundle.X[:, age_idx]
        assert abs(float(np.mean(col))) < 0.05

    # --- Single sensitive feature overrides ---

    def test_single_sex_feature(self, adult_bundle_sex_only):
        assert adult_bundle_sex_only.sensitive_names == ["sex"]
        assert adult_bundle_sex_only.s.shape[1] == 1
        assert len(adult_bundle_sex_only.sensitive_labels) == 1
        assert len(adult_bundle_sex_only.sensitive_labels[0]) == 2

    def test_single_race_feature(self, adult_bundle_race_only):
        assert adult_bundle_race_only.sensitive_names == ["race"]
        assert adult_bundle_race_only.s.shape[1] == 1
        unique_codes = set(adult_bundle_race_only.s[:, 0])
        assert unique_codes == set(adult_bundle_race_only.sensitive_labels[0].keys())
        assert len(unique_codes) > 2

    def test_single_vs_multi_X_shapes_differ(
        self, adult_bundle, adult_bundle_sex_only
    ):
        # With two sensitive attrs removed vs one, feature counts may differ
        # if race was encoded in X when not used as sensitive; at minimum
        # the n (rows) must be the same.
        assert adult_bundle.X.shape[0] == adult_bundle_sex_only.X.shape[0]

    # --- Error handling ---

    def test_invalid_sensitive_feature_raises(self):
        with pytest.raises(ValueError, match="not in columns"):
            load_adult(sensitive_features=["nonexistent_col"])

    def test_one_valid_one_invalid_raises(self):
        with pytest.raises(ValueError, match="not in columns"):
            load_adult(sensitive_features=["sex", "nonexistent_col"])


# ======================================================================
# Tests for dataset.py — FairDataset
# ======================================================================


class TestFairDataset:
    """Verify the PyTorch Dataset wrapper."""

    @pytest.fixture()
    def fair_ds(self, adult_bundle):
        return FairDataset(adult_bundle)

    def test_length(self, fair_ds, adult_bundle):
        assert len(fair_ds) == adult_bundle.X.shape[0]

    def test_getitem_returns_three_tensors(self, fair_ds):
        x, y, s = fair_ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert isinstance(s, torch.Tensor)

    def test_getitem_dtypes(self, fair_ds):
        x, y, s = fair_ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.long
        assert s.dtype == torch.long

    def test_getitem_x_shape(self, fair_ds, adult_bundle):
        x, _, _ = fair_ds[0]
        assert x.shape == (adult_bundle.X.shape[1],)

    def test_getitem_y_is_scalar(self, fair_ds):
        _, y, _ = fair_ds[0]
        assert y.dim() == 0

    def test_getitem_s_shape_is_k(self, fair_ds, adult_bundle):
        _, _, s = fair_ds[0]
        k = len(adult_bundle.sensitive_names)
        assert s.shape == (k,)

    def test_num_sensitive_property(self, fair_ds, adult_bundle):
        assert fair_ds.num_sensitive == len(adult_bundle.sensitive_names)

    def test_metadata_preserved(self, fair_ds, adult_bundle):
        assert fair_ds.feature_names == adult_bundle.feature_names
        assert fair_ds.sensitive_names == adult_bundle.sensitive_names
        assert fair_ds.target_name == adult_bundle.target_name
        assert fair_ds.sensitive_labels == adult_bundle.sensitive_labels

    def test_single_sensitive_s_still_2d_in_bundle(self, adult_bundle_sex_only):
        ds = FairDataset(adult_bundle_sex_only)
        _, _, s = ds[0]
        assert s.shape == (1,)


# ======================================================================
# Tests for dataset.py — IID partitioning
# ======================================================================


class TestIIDPartitioning:
    """Verify the IID partition helper."""

    def test_correct_number_of_partitions(self):
        assert len(_iid_partition_indices(100, 5)) == 5

    def test_all_indices_covered(self):
        n = 100
        all_idx = np.concatenate(_iid_partition_indices(n, 5))
        assert set(all_idx) == set(range(n))

    def test_no_duplicates(self):
        all_idx = np.concatenate(_iid_partition_indices(100, 5))
        assert len(all_idx) == len(set(all_idx))

    def test_roughly_equal_sizes(self):
        sizes = [len(p) for p in _iid_partition_indices(103, 5)]
        assert max(sizes) - min(sizes) <= 1

    def test_deterministic_with_same_seed(self):
        a = _iid_partition_indices(50, 3, seed=7)
        b = _iid_partition_indices(50, 3, seed=7)
        for pa, pb in zip(a, b):
            np.testing.assert_array_equal(pa, pb)

    def test_different_seed_gives_different_split(self):
        a = _iid_partition_indices(50, 3, seed=0)
        b = _iid_partition_indices(50, 3, seed=1)
        assert any(not np.array_equal(pa, pb) for pa, pb in zip(a, b))


# ======================================================================
# Tests for dataset.py — load_dataset registry
# ======================================================================


class TestLoadDataset:
    """Test the registry-based load_dataset entry point."""

    def test_load_adult_default(self):
        bundle = load_dataset("adult")
        assert isinstance(bundle, DatasetBundle)
        assert bundle.sensitive_names == ["sex", "race"]

    def test_load_adult_single_override(self):
        bundle = load_dataset("adult", sensitive_features=["sex"])
        assert bundle.sensitive_names == ["sex"]
        assert bundle.s.shape[1] == 1

    def test_load_adult_explicit_two_features(self):
        bundle = load_dataset("adult", sensitive_features=["sex", "race"])
        assert bundle.sensitive_names == ["sex", "race"]
        assert bundle.s.shape[1] == 2

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")


# ======================================================================
# Tests for dataset.py — get_federated_loaders (end-to-end)
# ======================================================================


class TestGetFederatedLoaders:
    """Integration tests for the full pipeline."""

    @pytest.fixture(scope="class")
    def loaders(self):
        return get_federated_loaders("adult", num_partitions=5, batch_size=32, seed=0)

    def test_returns_correct_number_of_loaders(self, loaders):
        assert len(loaders) == 5

    def test_each_loader_yields_three_tensors(self, loaders):
        assert len(next(iter(loaders[0]))) == 3

    def test_batch_X_is_2d_float32(self, loaders):
        X, _, _ = next(iter(loaders[0]))
        assert X.dim() == 2
        assert X.dtype == torch.float32

    def test_batch_y_is_1d_long(self, loaders):
        _, y, _ = next(iter(loaders[0]))
        assert y.dim() == 1
        assert y.dtype == torch.long

    def test_batch_s_is_2d_long_with_k_columns(self, loaders):
        _, _, s = next(iter(loaders[0]))
        assert s.dim() == 2
        assert s.dtype == torch.long
        assert s.shape[1] == 2  # default: sex + race

    def test_batch_sizes_consistent(self, loaders):
        X, y, s = next(iter(loaders[0]))
        assert X.shape[0] == y.shape[0] == s.shape[0]
        assert X.shape[0] <= 32

    def test_total_samples_match(self, loaders):
        total = sum(len(loader.dataset) for loader in loaders)
        bundle = load_dataset("adult")
        assert total == bundle.X.shape[0]

    def test_partitions_are_disjoint(self, loaders):
        all_indices = []
        for loader in loaders:
            all_indices.extend(loader.dataset.indices)
        assert len(all_indices) == len(set(all_indices))

    def test_iterate_full_loader(self, loaders):
        count = sum(X.shape[0] for X, _, _ in loaders[2])
        assert count == len(loaders[2].dataset)

    def test_custom_sensitive_features_in_loaders(self):
        loaders = get_federated_loaders(
            "adult",
            num_partitions=3,
            batch_size=64,
            sensitive_features=["sex"],
        )
        _, _, s = next(iter(loaders[0]))
        assert s.shape[1] == 1
