"""Tests for federated/secure_aggregation.py — SecureAggregator.

Covers creation, mask generation, aggregate with masks, HMAC-based
pair seed, mask cancellation verification, and mathematical correctness.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from tests.conftest import PROJECT_ROOT, load_module

mod = load_module("federated.secure_aggregation", "federated/secure_aggregation.py")

SecureAggregator = mod.SecureAggregator


class TestSecureAggregatorCreation:
    """Tests for SecureAggregator instantiation."""

    def test_creation_default_seed(self):
        sa = SecureAggregator()
        assert sa.seed == 0

    def test_creation_custom_seed(self):
        sa = SecureAggregator(seed=42)
        assert sa.seed == 42


class TestPairSeed:
    """Tests for deterministic pair seed generation."""

    def test_pair_seed_deterministic(self):
        sa = SecureAggregator(seed=0)
        s1 = sa._pair_seed(0, 1)
        s2 = sa._pair_seed(0, 1)
        assert s1 == s2

    def test_pair_seed_different_pairs(self):
        sa = SecureAggregator(seed=0)
        s01 = sa._pair_seed(0, 1)
        s02 = sa._pair_seed(0, 2)
        assert s01 != s02

    def test_pair_seed_different_base_seeds(self):
        sa1 = SecureAggregator(seed=0)
        sa2 = SecureAggregator(seed=999)
        s1 = sa1._pair_seed(0, 1)
        s2 = sa2._pair_seed(0, 1)
        assert s1 != s2

    def test_pair_seed_is_integer(self):
        sa = SecureAggregator(seed=0)
        s = sa._pair_seed(3, 7)
        assert isinstance(s, int)


class TestMaskGeneration:
    """Tests for pairwise mask generation and cancellation."""

    def test_masks_cancel_two_clients(self):
        sa = SecureAggregator(seed=0)
        template = [np.ones((3, 4)), np.ones((4,))]
        masks = sa._generate_pairwise_masks(2, template)
        assert len(masks) == 2
        assert SecureAggregator.verify_mask_cancellation(masks)

    def test_masks_cancel_five_clients(self):
        sa = SecureAggregator(seed=0)
        template = [np.ones((10, 5)), np.ones((5,))]
        masks = sa._generate_pairwise_masks(5, template)
        assert len(masks) == 5
        assert SecureAggregator.verify_mask_cancellation(masks)

    def test_masks_cancel_single_client(self):
        sa = SecureAggregator(seed=0)
        template = [np.ones((2, 2))]
        masks = sa._generate_pairwise_masks(1, template)
        assert len(masks) == 1
        # Single client: mask should be all zeros
        npt.assert_allclose(masks[0][0], np.zeros((2, 2)), atol=1e-10)

    def test_mask_shapes_match_template(self):
        sa = SecureAggregator(seed=0)
        template = [np.zeros((7, 3)), np.zeros((3,))]
        masks = sa._generate_pairwise_masks(3, template)
        for client_masks in masks:
            assert client_masks[0].shape == (7, 3)
            assert client_masks[1].shape == (3,)

    def test_verify_mask_cancellation_bad_masks(self):
        bad_masks = [
            [np.array([1.0, 2.0])],
            [np.array([3.0, 4.0])],
        ]
        assert not SecureAggregator.verify_mask_cancellation(bad_masks)


class TestSecureAggregate:
    """Tests for the full secure aggregation pipeline."""

    def test_aggregate_equals_weighted_average(self):
        sa = SecureAggregator(seed=0)
        updates = [
            [np.array([1.0, 2.0]), np.array([3.0])],
            [np.array([5.0, 6.0]), np.array([7.0])],
        ]
        weights = [0.5, 0.5]
        result = sa.aggregate(updates, weights)
        # Expected: weighted average
        npt.assert_allclose(result[0], [3.0, 4.0], atol=1e-8)
        npt.assert_allclose(result[1], [5.0], atol=1e-8)

    def test_aggregate_unequal_weights(self):
        sa = SecureAggregator(seed=0)
        updates = [
            [np.array([0.0, 0.0])],
            [np.array([10.0, 10.0])],
        ]
        weights = [0.3, 0.7]
        result = sa.aggregate(updates, weights)
        npt.assert_allclose(result[0], [7.0, 7.0], atol=1e-8)

    def test_aggregate_single_client(self):
        sa = SecureAggregator(seed=0)
        updates = [[np.array([4.0, 5.0])]]
        weights = [1.0]
        result = sa.aggregate(updates, weights)
        npt.assert_allclose(result[0], [4.0, 5.0], atol=1e-8)

    def test_aggregate_preserves_shapes(self):
        sa = SecureAggregator(seed=42)
        updates = [
            [np.random.randn(3, 4), np.random.randn(4)],
            [np.random.randn(3, 4), np.random.randn(4)],
            [np.random.randn(3, 4), np.random.randn(4)],
        ]
        weights = [1 / 3, 1 / 3, 1 / 3]
        result = sa.aggregate(updates, weights)
        assert result[0].shape == (3, 4)
        assert result[1].shape == (4,)

    def test_aggregate_deterministic(self):
        sa = SecureAggregator(seed=7)
        updates = [
            [np.array([1.0, 2.0])],
            [np.array([3.0, 4.0])],
        ]
        weights = [0.5, 0.5]
        r1 = sa.aggregate(updates, weights)
        r2 = sa.aggregate(updates, weights)
        npt.assert_allclose(r1[0], r2[0])
