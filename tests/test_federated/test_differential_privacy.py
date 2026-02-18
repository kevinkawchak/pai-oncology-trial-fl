"""Tests for federated/differential_privacy.py — DifferentialPrivacy.

Covers creation, noise addition, gradient clipping, zero-norm handling,
budget tracking, epsilon computation, and mathematical correctness.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

from tests.conftest import PROJECT_ROOT, load_module

mod = load_module("federated.differential_privacy", "federated/differential_privacy.py")

DifferentialPrivacy = mod.DifferentialPrivacy


class TestDifferentialPrivacyCreation:
    """Tests for DP mechanism instantiation."""

    def test_creation_defaults(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        assert dp.epsilon == 1.0
        assert dp.delta == 1e-5
        assert dp.max_grad_norm == 1.0
        assert dp.rounds_completed == 0

    def test_creation_custom_noise_multiplier(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, noise_multiplier=2.0)
        assert dp.noise_multiplier == 2.0

    def test_auto_calibrated_noise_multiplier(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        expected = math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 1.0
        npt.assert_allclose(dp.noise_multiplier, expected, rtol=1e-6)

    def test_epsilon_zero_raises(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            DifferentialPrivacy(epsilon=0.0, delta=1e-5)

    def test_epsilon_negative_raises(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            DifferentialPrivacy(epsilon=-1.0, delta=1e-5)

    def test_delta_zero_raises(self):
        with pytest.raises(ValueError, match="delta must be in"):
            DifferentialPrivacy(epsilon=1.0, delta=0.0)

    def test_delta_one_raises(self):
        with pytest.raises(ValueError, match="delta must be in"):
            DifferentialPrivacy(epsilon=1.0, delta=1.0)

    def test_delta_negative_raises(self):
        with pytest.raises(ValueError, match="delta must be in"):
            DifferentialPrivacy(epsilon=1.0, delta=-0.1)


class TestGradientClipping:
    """Tests for parameter clipping."""

    def test_clip_within_norm_unchanged(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=10.0)
        params = [np.array([1.0, 0.0]), np.array([0.0])]
        clipped = dp.clip_parameters(params)
        npt.assert_allclose(clipped[0], [1.0, 0.0])
        npt.assert_allclose(clipped[1], [0.0])

    def test_clip_exceeding_norm_scales(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        params = [np.array([3.0, 4.0])]  # L2 norm = 5
        clipped = dp.clip_parameters(params)
        total_norm = np.sqrt(sum(float(np.sum(p**2)) for p in clipped))
        npt.assert_allclose(total_norm, 1.0, atol=1e-6)

    def test_clip_preserves_direction(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        params = [np.array([3.0, 4.0])]
        clipped = dp.clip_parameters(params)
        # Direction should be the same: ratio of elements preserved
        npt.assert_allclose(clipped[0][0] / clipped[0][1], 3.0 / 4.0, atol=1e-6)

    def test_clip_zero_norm_returns_copy(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        params = [np.array([0.0, 0.0])]
        clipped = dp.clip_parameters(params)
        npt.assert_allclose(clipped[0], [0.0, 0.0])
        # Ensure it is a copy
        clipped[0][0] = 999.0
        assert params[0][0] == 0.0

    def test_clip_exactly_at_norm(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=5.0)
        params = [np.array([3.0, 4.0])]  # norm exactly 5
        clipped = dp.clip_parameters(params)
        npt.assert_allclose(clipped[0], [3.0, 4.0], atol=1e-6)

    def test_clip_multiple_tensors(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        # Combined norm = sqrt(1 + 4 + 4) = 3
        params = [np.array([1.0]), np.array([2.0]), np.array([2.0])]
        clipped = dp.clip_parameters(params)
        total_norm = math.sqrt(sum(float(np.sum(p**2)) for p in clipped))
        npt.assert_allclose(total_norm, 1.0, atol=1e-6)


class TestNoiseAddition:
    """Tests for noise addition to parameters."""

    def test_add_noise_changes_parameters(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        params = [np.ones((5, 3))]
        noisy = dp.add_noise_to_parameters(params, num_clients=1, seed=42)
        assert not np.allclose(noisy[0], params[0])

    def test_add_noise_preserves_shape(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        params = [np.ones((3, 4)), np.zeros(4)]
        noisy = dp.add_noise_to_parameters(params, num_clients=2, seed=0)
        assert noisy[0].shape == (3, 4)
        assert noisy[1].shape == (4,)

    def test_add_noise_deterministic_with_seed(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        params = [np.ones((5,))]
        noisy1 = dp.add_noise_to_parameters(params, seed=123)
        # Reset rounds counter for fair comparison
        dp2 = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        noisy2 = dp2.add_noise_to_parameters(params, seed=123)
        npt.assert_allclose(noisy1[0], noisy2[0])

    def test_noise_scales_with_sensitivity(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        params = [np.zeros((1000,))]
        # More clients => lower sensitivity => less noise
        noisy_1 = dp.add_noise_to_parameters(params, num_clients=1, seed=42)
        dp2 = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        noisy_100 = dp2.add_noise_to_parameters(params, num_clients=100, seed=42)
        var_1 = np.var(noisy_1[0])
        var_100 = np.var(noisy_100[0])
        assert var_1 > var_100


class TestBudgetTracking:
    """Tests for privacy budget accounting."""

    def test_initial_budget_zero(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        spent = dp.get_privacy_spent()
        assert spent["total_epsilon_spent"] == 0.0
        assert spent["rounds_completed"] == 0

    def test_budget_increments_per_round(self):
        dp = DifferentialPrivacy(epsilon=0.5, delta=1e-5)
        dp.add_noise_to_parameters([np.ones(3)])
        dp.add_noise_to_parameters([np.ones(3)])
        spent = dp.get_privacy_spent()
        npt.assert_allclose(spent["total_epsilon_spent"], 1.0)
        assert spent["rounds_completed"] == 2

    def test_estimate_remaining_rounds(self):
        dp = DifferentialPrivacy(epsilon=0.5, delta=1e-5)
        remaining = dp.estimate_remaining_rounds(total_budget=5.0)
        assert remaining == 10

    def test_estimate_remaining_rounds_after_spending(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        dp.add_noise_to_parameters([np.ones(3)])
        dp.add_noise_to_parameters([np.ones(3)])
        remaining = dp.estimate_remaining_rounds(total_budget=5.0)
        assert remaining == 3

    def test_estimate_remaining_rounds_budget_exhausted(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        for _ in range(5):
            dp.add_noise_to_parameters([np.ones(3)])
        remaining = dp.estimate_remaining_rounds(total_budget=5.0)
        assert remaining == 0

    def test_privacy_spent_includes_multiplier(self):
        dp = DifferentialPrivacy(epsilon=2.0, delta=1e-5, noise_multiplier=3.0)
        spent = dp.get_privacy_spent()
        assert spent["noise_multiplier"] == 3.0
        assert spent["epsilon_per_round"] == 2.0
        assert spent["delta_per_round"] == 1e-5
