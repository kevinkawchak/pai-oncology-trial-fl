"""Federation coordinator — orchestrates multi-site training rounds.

The coordinator manages the global model, distributes parameters to
participating clients, collects updates, and produces an aggregated
model each round (Federated Averaging).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from federated.differential_privacy import DifferentialPrivacy
from federated.model import FederatedModel, ModelConfig
from federated.secure_aggregation import SecureAggregator

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Metrics recorded after a single federation round."""

    round_number: int
    num_clients: int
    global_metrics: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


class FederationCoordinator:
    """Orchestrates federated training across multiple client sites.

    Implements Federated Averaging (FedAvg) with optional secure
    aggregation and differential privacy.

    Args:
        model_config: Architecture specification for the global model.
        num_rounds: Total training rounds to execute.
        min_clients: Minimum clients required to proceed with a round.
        use_secure_aggregation: Enable mask-based secure aggregation.
        use_differential_privacy: Enable Gaussian-mechanism DP.
        dp_epsilon: Privacy budget epsilon (if DP enabled).
        dp_delta: Privacy budget delta (if DP enabled).
        dp_max_grad_norm: Maximum gradient norm for DP clipping.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        num_rounds: int = 10,
        min_clients: int = 2,
        use_secure_aggregation: bool = False,
        use_differential_privacy: bool = False,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        dp_max_grad_norm: float = 1.0,
    ):
        self.model_config = model_config
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.global_model: FederatedModel | None = None
        self.round_history: list[RoundResult] = []
        self.current_round = 0

        self.secure_aggregator: SecureAggregator | None = None
        if use_secure_aggregation:
            self.secure_aggregator = SecureAggregator()

        self.dp: DifferentialPrivacy | None = None
        if use_differential_privacy:
            self.dp = DifferentialPrivacy(
                epsilon=dp_epsilon,
                delta=dp_delta,
                max_grad_norm=dp_max_grad_norm,
            )

    def initialize(self) -> list[np.ndarray]:
        """Initialize the global model and return its parameters."""
        self.global_model = FederatedModel.from_config(self.model_config)
        self.current_round = 0
        self.round_history = []
        logger.info("Federation initialized with model config: %s", self.model_config)
        return self.global_model.get_parameters()

    def get_global_parameters(self) -> list[np.ndarray]:
        """Return current global model parameters."""
        if self.global_model is None:
            raise RuntimeError("Coordinator not initialized. Call initialize() first.")
        return self.global_model.get_parameters()

    def run_round(
        self,
        client_updates: list[list[np.ndarray]],
        client_sample_counts: list[int] | None = None,
        eval_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> RoundResult:
        """Execute one federation round: aggregate client updates into a new global model.

        Args:
            client_updates: List of parameter lists, one per client.
            client_sample_counts: Number of training samples at each client
                (used for weighted averaging). If None, uniform weighting.
            eval_data: Optional (X, y_labels) tuple for evaluating the
                global model after aggregation.

        Returns:
            RoundResult with round metrics.

        Raises:
            RuntimeError: If coordinator is not initialized.
            ValueError: If fewer than min_clients updates are provided.
        """
        if self.global_model is None:
            raise RuntimeError("Coordinator not initialized. Call initialize() first.")

        if len(client_updates) < self.min_clients:
            raise ValueError(
                f"Received {len(client_updates)} client updates, but minimum is {self.min_clients}."
            )

        start = time.time()
        self.current_round += 1

        # Compute client weights
        if client_sample_counts is not None:
            total = sum(client_sample_counts)
            weights = [c / total for c in client_sample_counts]
        else:
            weights = [1.0 / len(client_updates)] * len(client_updates)

        # Aggregate
        if self.secure_aggregator is not None:
            aggregated = self.secure_aggregator.aggregate(client_updates, weights)
        else:
            aggregated = self._fedavg(client_updates, weights)

        # Apply differential privacy noise
        if self.dp is not None:
            aggregated = self.dp.add_noise_to_parameters(
                aggregated, num_clients=len(client_updates)
            )

        self.global_model.set_parameters(aggregated)

        # Evaluate
        metrics: dict[str, float] = {}
        if eval_data is not None:
            x_eval, y_eval = eval_data
            metrics = self.global_model.evaluate(x_eval, y_eval)

        elapsed = time.time() - start
        result = RoundResult(
            round_number=self.current_round,
            num_clients=len(client_updates),
            global_metrics=metrics,
            elapsed_seconds=elapsed,
        )
        self.round_history.append(result)

        logger.info(
            "Round %d complete — %d clients, metrics=%s",
            self.current_round,
            len(client_updates),
            metrics,
        )
        return result

    @staticmethod
    def _fedavg(updates: list[list[np.ndarray]], weights: list[float]) -> list[np.ndarray]:
        """Federated Averaging: weighted mean of client parameter sets."""
        num_params = len(updates[0])
        aggregated: list[np.ndarray] = []
        for p in range(num_params):
            weighted_sum = sum(w * updates[c][p] for c, w in enumerate(weights))
            aggregated.append(weighted_sum)
        return aggregated

    def get_training_summary(self) -> dict:
        """Return a summary of all completed rounds."""
        return {
            "total_rounds": self.current_round,
            "rounds": [
                {
                    "round": r.round_number,
                    "clients": r.num_clients,
                    "metrics": r.global_metrics,
                    "elapsed_s": round(r.elapsed_seconds, 4),
                }
                for r in self.round_history
            ],
        }
