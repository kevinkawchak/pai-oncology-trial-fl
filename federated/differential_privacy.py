"""Differential privacy for federated learning.

Implements the Gaussian mechanism: model updates are clipped to a
maximum L2 norm and calibrated Gaussian noise is added, providing
(epsilon, delta)-differential privacy guarantees per round.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """Gaussian-mechanism differential privacy for model updates.

    Args:
        epsilon: Privacy budget per round.
        delta: Failure probability per round.
        max_grad_norm: Maximum L2 norm for gradient clipping.
        noise_multiplier: If provided, overrides the automatically
            computed multiplier derived from epsilon/delta.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float | None = None,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")

        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.rounds_completed = 0
        self._total_epsilon_spent = 0.0

        if noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
        else:
            self.noise_multiplier = self._calibrate_noise()

    def _calibrate_noise(self) -> float:
        """Compute noise multiplier from the analytic Gaussian mechanism."""
        return math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon

    def clip_parameters(self, params: list[np.ndarray]) -> list[np.ndarray]:
        """Clip parameter tensors so their total L2 norm <= max_grad_norm."""
        total_norm = math.sqrt(sum(float(np.sum(p**2)) for p in params))
        scale = min(1.0, self.max_grad_norm / max(total_norm, 1e-12))
        return [p * scale for p in params]

    def add_noise_to_parameters(
        self,
        params: list[np.ndarray],
        num_clients: int = 1,
        seed: int | None = None,
    ) -> list[np.ndarray]:
        """Add calibrated Gaussian noise to aggregated parameters.

        Args:
            params: Aggregated parameter tensors.
            num_clients: Number of clients in the aggregation (used
                for sensitivity scaling).
            seed: Optional seed for reproducible noise.

        Returns:
            Noisy parameter list.
        """
        rng = np.random.default_rng(seed)
        sensitivity = self.max_grad_norm / max(num_clients, 1)
        noise_std = self.noise_multiplier * sensitivity

        noisy_params = []
        for p in params:
            noise = rng.normal(0, noise_std, size=p.shape)
            noisy_params.append(p + noise)

        self.rounds_completed += 1
        # Simple composition: linear accumulation (conservative bound)
        self._total_epsilon_spent += self.epsilon

        logger.info(
            "DP noise applied — sigma=%.6f, eps_spent=%.4f",
            noise_std,
            self._total_epsilon_spent,
        )
        return noisy_params

    def get_privacy_spent(self) -> dict[str, float]:
        """Return cumulative privacy expenditure."""
        return {
            "epsilon_per_round": self.epsilon,
            "delta_per_round": self.delta,
            "rounds_completed": self.rounds_completed,
            "total_epsilon_spent": self._total_epsilon_spent,
            "noise_multiplier": self.noise_multiplier,
        }

    def estimate_remaining_rounds(self, total_budget: float) -> int:
        """Estimate how many more rounds can run within a total budget."""
        remaining = total_budget - self._total_epsilon_spent
        if remaining <= 0:
            return 0
        return int(remaining / self.epsilon)
