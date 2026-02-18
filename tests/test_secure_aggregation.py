"""Tests for secure aggregation.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.
"""

import numpy as np

from federated.secure_aggregation import SecureAggregator


class TestSecureAggregator:
    def test_aggregate_matches_plaintext(self):
        """Secure aggregation should produce the same result as plain averaging."""
        agg = SecureAggregator(seed=42)

        # Create 3 client updates
        updates = [
            [np.array([1.0, 2.0]), np.array([3.0])],
            [np.array([4.0, 5.0]), np.array([6.0])],
            [np.array([7.0, 8.0]), np.array([9.0])],
        ]
        weights = [1.0 / 3, 1.0 / 3, 1.0 / 3]

        result = agg.aggregate(updates, weights)

        # Expected: weighted average
        expected_0 = (1.0 + 4.0 + 7.0) / 3
        expected_1 = (2.0 + 5.0 + 8.0) / 3
        expected_2 = (3.0 + 6.0 + 9.0) / 3

        assert np.allclose(result[0], [expected_0, expected_1], atol=1e-6)
        assert np.allclose(result[1], [expected_2], atol=1e-6)

    def test_mask_cancellation(self):
        """Pairwise masks should sum to zero across all clients."""
        agg = SecureAggregator(seed=0)
        template = [np.zeros((3, 4)), np.zeros(4)]
        masks = agg._generate_pairwise_masks(5, template)
        assert SecureAggregator.verify_mask_cancellation(masks)

    def test_two_clients(self):
        """Aggregation with exactly two clients."""
        agg = SecureAggregator()
        updates = [
            [np.array([10.0, 20.0])],
            [np.array([30.0, 40.0])],
        ]
        weights = [0.5, 0.5]
        result = agg.aggregate(updates, weights)
        assert np.allclose(result[0], [20.0, 30.0], atol=1e-6)

    def test_unequal_weights(self):
        """Weighted aggregation with different client weights."""
        agg = SecureAggregator()
        updates = [
            [np.array([0.0])],
            [np.array([10.0])],
        ]
        weights = [0.25, 0.75]
        result = agg.aggregate(updates, weights)
        assert np.allclose(result[0], [7.5], atol=1e-6)

    def test_deterministic_with_same_seed(self):
        """Same seed produces same masks."""
        agg1 = SecureAggregator(seed=123)
        agg2 = SecureAggregator(seed=123)
        template = [np.zeros((2, 3))]
        masks1 = agg1._generate_pairwise_masks(3, template)
        masks2 = agg2._generate_pairwise_masks(3, template)
        for m1, m2 in zip(masks1, masks2):
            for p1, p2 in zip(m1, m2):
                assert np.allclose(p1, p2)
