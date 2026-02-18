"""Tests for differential privacy mechanisms.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.
"""

import numpy as np
import pytest

from federated.differential_privacy import DifferentialPrivacy


class TestDifferentialPrivacy:
    def test_init(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        assert dp.epsilon == 1.0
        assert dp.delta == 1e-5
        assert dp.noise_multiplier > 0

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            DifferentialPrivacy(epsilon=-1.0)

    def test_invalid_delta(self):
        with pytest.raises(ValueError, match="delta"):
            DifferentialPrivacy(epsilon=1.0, delta=0.0)

    def test_clip_parameters(self):
        dp = DifferentialPrivacy(max_grad_norm=1.0)
        params = [np.array([10.0, 0.0, 0.0])]
        clipped = dp.clip_parameters(params)
        total_norm = np.sqrt(sum(float(np.sum(p**2)) for p in clipped))
        assert total_norm <= 1.0 + 1e-6

    def test_clip_within_norm_unchanged(self):
        dp = DifferentialPrivacy(max_grad_norm=100.0)
        params = [np.array([1.0, 2.0])]
        clipped = dp.clip_parameters(params)
        assert np.allclose(clipped[0], params[0])

    def test_add_noise_changes_parameters(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        params = [np.zeros(10)]
        noisy = dp.add_noise_to_parameters(params, num_clients=1, seed=42)
        # Noise should have been added
        assert not np.allclose(noisy[0], params[0])

    def test_add_noise_is_reproducible_with_seed(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        params = [np.ones(5)]
        n1 = dp.add_noise_to_parameters(params, seed=42)
        dp2 = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        n2 = dp2.add_noise_to_parameters(params, seed=42)
        assert np.allclose(n1[0], n2[0])

    def test_privacy_budget_tracking(self):
        dp = DifferentialPrivacy(epsilon=0.5, delta=1e-5)
        params = [np.ones(5)]
        dp.add_noise_to_parameters(params)
        dp.add_noise_to_parameters(params)
        spent = dp.get_privacy_spent()
        assert spent["rounds_completed"] == 2
        assert spent["total_epsilon_spent"] == 1.0

    def test_estimate_remaining_rounds(self):
        dp = DifferentialPrivacy(epsilon=0.5, delta=1e-5)
        remaining = dp.estimate_remaining_rounds(total_budget=5.0)
        assert remaining == 10

    def test_noisy_aggregation_still_reasonable(self):
        """Adding DP noise to aggregated parameters should still yield
        a model that produces valid outputs."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)

        from federated.model import FederatedModel

        model = FederatedModel(input_dim=10, hidden_dims=[8], output_dim=2, seed=42)
        params = model.get_parameters()
        noisy_params = dp.add_noise_to_parameters(params, num_clients=5, seed=0)
        model.set_parameters(noisy_params)

        x = np.random.randn(10, 10)
        out = model.forward(x)
        # Model should still produce valid probabilities
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(out >= 0)
