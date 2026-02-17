"""Tests for data ingestion and partitioning."""

import numpy as np
import pytest

from federated.data_ingestion import DataPartitioner, generate_synthetic_oncology_data


class TestGenerateSyntheticData:
    def test_default_shape(self):
        x, y = generate_synthetic_oncology_data()
        assert x.shape == (1000, 30)
        assert y.shape == (1000,)

    def test_custom_shape(self):
        x, y = generate_synthetic_oncology_data(n_samples=200, n_features=10, n_classes=3)
        assert x.shape == (200, 10)
        assert set(np.unique(y)).issubset({0, 1, 2})

    def test_class_balance(self):
        x, y = generate_synthetic_oncology_data(
            n_samples=1000, n_classes=2, class_balance=[0.3, 0.7]
        )
        counts = np.bincount(y)
        assert counts[0] < counts[1]

    def test_reproducibility(self):
        x1, y1 = generate_synthetic_oncology_data(seed=42)
        x2, y2 = generate_synthetic_oncology_data(seed=42)
        assert np.allclose(x1, x2)
        assert np.array_equal(y1, y2)


class TestDataPartitioner:
    def test_iid_partition(self):
        x, y = generate_synthetic_oncology_data(n_samples=200, seed=0)
        partitioner = DataPartitioner(num_sites=4, strategy="iid")
        sites = partitioner.partition(x, y)
        assert len(sites) == 4
        total = sum(s.num_train + s.num_test for s in sites)
        assert total == 200

    def test_non_iid_partition(self):
        x, y = generate_synthetic_oncology_data(n_samples=200, seed=0)
        partitioner = DataPartitioner(num_sites=3, strategy="non_iid")
        sites = partitioner.partition(x, y)
        assert len(sites) == 3

    def test_train_test_split(self):
        x, y = generate_synthetic_oncology_data(n_samples=100, seed=0)
        partitioner = DataPartitioner(num_sites=2, test_fraction=0.2)
        sites = partitioner.partition(x, y)
        for site in sites:
            total = site.num_train + site.num_test
            assert site.num_test >= 1
            assert site.num_train >= 1
            assert total > 0

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            DataPartitioner(strategy="invalid")

    def test_invalid_num_sites(self):
        with pytest.raises(ValueError, match="num_sites"):
            DataPartitioner(num_sites=0)

    def test_site_ids(self):
        x, y = generate_synthetic_oncology_data(n_samples=100, seed=0)
        partitioner = DataPartitioner(num_sites=3)
        sites = partitioner.partition(x, y)
        ids = [s.site_id for s in sites]
        assert ids == ["site_0", "site_1", "site_2"]
