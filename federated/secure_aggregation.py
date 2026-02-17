"""Secure aggregation for federated learning.

Implements a simplified mask-based secure aggregation protocol so
that the coordinator only observes the *aggregate* of client updates,
not any individual client's contribution.

Protocol overview
-----------------
1. Each pair of clients (i, j) agrees on a shared random seed.
2. Client i adds mask_ij to its update; client j subtracts mask_ij.
3. When the coordinator sums all masked updates the masks cancel,
   yielding the true aggregate — without revealing individual updates.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np

logger = logging.getLogger(__name__)


class SecureAggregator:
    """Mask-based secure aggregation.

    Args:
        seed: Base seed for reproducible mask generation.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def aggregate(
        self,
        client_updates: list[list[np.ndarray]],
        weights: list[float],
    ) -> list[np.ndarray]:
        """Securely aggregate client parameter updates.

        In a production deployment the masking happens *at* each client.
        Here we simulate the full protocol server-side for demonstration
        and testing purposes.

        Args:
            client_updates: Parameter lists from each client.
            weights: Per-client averaging weights (must sum to 1).

        Returns:
            Aggregated parameter list.
        """
        num_clients = len(client_updates)
        num_params = len(client_updates[0])

        # Step 1 — generate pairwise masks
        masks = self._generate_pairwise_masks(num_clients, client_updates[0])

        # Step 2 — apply masks to each client's weighted update
        masked_updates: list[list[np.ndarray]] = []
        for c in range(num_clients):
            masked = []
            for p in range(num_params):
                val = weights[c] * client_updates[c][p] + masks[c][p]
                masked.append(val)
            masked_updates.append(masked)

        # Step 3 — sum masked updates (masks cancel out)
        aggregated: list[np.ndarray] = []
        for p in range(num_params):
            total = sum(masked_updates[c][p] for c in range(num_clients))
            aggregated.append(total)

        logger.info(
            "Secure aggregation complete — %d clients, %d parameters",
            num_clients,
            num_params,
        )
        return aggregated

    def _generate_pairwise_masks(
        self,
        num_clients: int,
        param_template: list[np.ndarray],
    ) -> list[list[np.ndarray]]:
        """Generate zero-sum masks for all client pairs.

        For each ordered pair (i, j) with i < j a random mask is
        generated.  Client i adds the mask; client j subtracts it.
        The net contribution across all clients is zero for every
        parameter tensor.
        """
        num_params = len(param_template)
        masks: list[list[np.ndarray]] = [
            [np.zeros_like(p) for p in param_template] for _ in range(num_clients)
        ]

        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                seed = self._pair_seed(i, j)
                rng = np.random.default_rng(seed)
                for p in range(num_params):
                    mask = rng.normal(0, 0.01, size=param_template[p].shape)
                    masks[i][p] = masks[i][p] + mask
                    masks[j][p] = masks[j][p] - mask

        return masks

    def _pair_seed(self, i: int, j: int) -> int:
        """Deterministic seed for a client pair."""
        data = f"{self.seed}-{i}-{j}".encode()
        return int(hashlib.sha256(data).hexdigest()[:8], 16)

    @staticmethod
    def verify_mask_cancellation(
        masks: list[list[np.ndarray]],
    ) -> bool:
        """Verify that masks sum to zero (for testing)."""
        num_params = len(masks[0])
        for p in range(num_params):
            total = sum(m[p] for m in masks)
            if not np.allclose(total, 0, atol=1e-10):
                return False
        return True
