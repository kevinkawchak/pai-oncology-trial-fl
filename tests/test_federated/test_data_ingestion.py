"""Tests for federated/data_ingestion.py — data generation and partitioning.

Covers generate_synthetic_data, DataPartitioner, IID/non-IID strategies,
feature shapes, label distribution, edge cases, and SiteData properties.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from tests.conftest import PROJECT_ROOT, load_module

mod = load_module("federated.data_ingestion", "federated/data_ingestion.py")

generate_synthetic_oncology_data = mod.generate_synthetic_oncology_data
DataPartitioner = mod.DataPartitioner
SiteData = mod.SiteData


class TestGenerateSyntheticData:
    """Tests for synthetic data generation."""

    def test_default_generation(self):
        x, y = generate_synthetic_oncology_data()
        assert x.shape == (1000, 30)
        assert y.shape == (1000,)

    def test_custom_dimensions(self):
        x, y = generate_synthetic_oncology_data(n_samples=200, n_features=10, n_classes=3)
        assert x.shape == (200, 10)
        assert y.shape == (200,)
        assert set(np.unique(y)).issubset({0, 1, 2})

    def test_deterministic_with_seed(self):
        x1, y1 = generate_synthetic_oncology_data(seed=7)
        x2, y2 = generate_synthetic_oncology_data(seed=7)
        npt.assert_allclose(x1, x2)
        npt.assert_allclose(y1, y2)

    def test_different_seeds_differ(self):
        x1, _ = generate_synthetic_oncology_data(seed=0)
        x2, _ = generate_synthetic_oncology_data(seed=99)
        assert not np.allclose(x1, x2)

    def test_class_balance_uniform(self):
        x, y = generate_synthetic_oncology_data(n_samples=100, n_classes=2, seed=0)
        counts = np.bincount(y)
        assert len(counts) == 2
        # Uniform: roughly 50/50
        assert all(c > 30 for c in counts)

    def test_class_balance_custom(self):
        x, y = generate_synthetic_oncology_data(n_samples=1000, n_classes=2, class_balance=[0.8, 0.2], seed=0)
        counts = np.bincount(y)
        # Class 0 should dominate
        assert counts[0] > counts[1]

    def test_single_sample(self):
        x, y = generate_synthetic_oncology_data(n_samples=1, n_features=5, n_classes=2)
        assert x.shape == (1, 5)
        assert y.shape == (1,)


class TestDataPartitioner:
    """Tests for DataPartitioner."""

    def test_creation_defaults(self):
        dp = DataPartitioner()
        assert dp.num_sites == 2
        assert dp.strategy == "iid"
        assert dp.test_fraction == 0.2

    def test_invalid_num_sites_raises(self):
        with pytest.raises(ValueError, match="num_sites must be >= 1"):
            DataPartitioner(num_sites=0)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy must be"):
            DataPartitioner(strategy="weird")

    def test_iid_partition_num_sites(self):
        x, y = generate_synthetic_oncology_data(n_samples=100, n_features=5, seed=0)
        dp = DataPartitioner(num_sites=4, strategy="iid", seed=0)
        sites = dp.partition(x, y)
        assert len(sites) == 4

    def test_iid_partition_covers_all_samples(self):
        x, y = generate_synthetic_oncology_data(n_samples=100, n_features=5, seed=0)
        dp = DataPartitioner(num_sites=3, strategy="iid", test_fraction=0.2, seed=0)
        sites = dp.partition(x, y)
        total = sum(s.num_train + s.num_test for s in sites)
        assert total == 100

    def test_non_iid_partition_num_sites(self):
        x, y = generate_synthetic_oncology_data(n_samples=200, n_features=5, seed=0)
        dp = DataPartitioner(num_sites=3, strategy="non_iid", seed=0)
        sites = dp.partition(x, y)
        assert len(sites) == 3

    def test_site_data_properties(self):
        x, y = generate_synthetic_oncology_data(n_samples=100, n_features=5, seed=0)
        dp = DataPartitioner(num_sites=2, strategy="iid", seed=0)
        sites = dp.partition(x, y)
        site = sites[0]
        assert isinstance(site, SiteData)
        assert site.num_train == len(site.x_train)
        assert site.num_test == len(site.x_test)
        assert site.x_train.shape[1] == 5
        assert site.x_test.shape[1] == 5

    def test_single_site_partition(self):
        x, y = generate_synthetic_oncology_data(n_samples=50, n_features=3, seed=0)
        dp = DataPartitioner(num_sites=1, strategy="iid", seed=0)
        sites = dp.partition(x, y)
        assert len(sites) == 1
        assert sites[0].num_train + sites[0].num_test == 50
