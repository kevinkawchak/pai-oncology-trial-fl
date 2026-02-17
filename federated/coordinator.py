"""Federation coordinator — orchestrates multi-site training rounds.

The coordinator manages the global model, distributes parameters to
participating clients, collects updates, and produces an aggregated
model each round.  Supports three aggregation strategies:

- **FedAvg** (McMahan et al., 2017) — weighted parameter averaging.
- **FedProx** (Li et al., 2020) — adds a proximal term to handle
  heterogeneous client data distributions.
- **SCAFFOLD** (Karimireddy et al., 2020) — uses control variates to
  correct client drift in non-IID settings.

These strategies are selected via the ``strategy`` parameter and can be
combined with differential privacy and/or secure aggregation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from federated.differential_privacy import DifferentialPrivacy
from federated.model import FederatedModel, ModelConfig
from federated.secure_aggregation import SecureAggregator

logger = logging.getLogger(__name__)


class AggregationStrategy(str, Enum):
    """Supported aggregation strategies for federated learning."""

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"


@dataclass
class RoundResult:
    """Metrics recorded after a single federation round."""

    round_number: int
    num_clients: int
    global_metrics: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    converged: bool = False


class FederationCoordinator:
    """Orchestrates federated training across multiple client sites.

    Supports FedAvg, FedProx, and SCAFFOLD aggregation strategies with
    optional secure aggregation and differential privacy.  Includes
    automatic convergence detection based on rolling accuracy history.

    Args:
        model_config: Architecture specification for the global model.
        num_rounds: Total training rounds to execute.
        min_clients: Minimum clients required to proceed with a round.
        strategy: Aggregation strategy (``"fedavg"``, ``"fedprox"``,
            ``"scaffold"``).
        mu: Proximal term coefficient for FedProx.  Ignored for other
            strategies.
        use_secure_aggregation: Enable mask-based secure aggregation.
        use_differential_privacy: Enable Gaussian-mechanism DP.
        dp_epsilon: Privacy budget epsilon (if DP enabled).
        dp_delta: Privacy budget delta (if DP enabled).
        dp_max_grad_norm: Maximum gradient norm for DP clipping.
        convergence_window: Number of rounds to consider when checking
            for convergence.  Set to ``0`` to disable.
        convergence_threshold: Minimum accuracy improvement over the
            convergence window to be considered non-converged.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        num_rounds: int = 10,
        min_clients: int = 2,
        strategy: AggregationStrategy | str = AggregationStrategy.FEDAVG,
        mu: float = 0.01,
        use_secure_aggregation: bool = False,
        use_differential_privacy: bool = False,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        dp_max_grad_norm: float = 1.0,
        convergence_window: int = 5,
        convergence_threshold: float = 0.001,
    ):
        self.model_config = model_config
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.mu = mu
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold

        if isinstance(strategy, str):
            strategy = AggregationStrategy(strategy)
        self.strategy = strategy

        self.global_model: FederatedModel | None = None
        self.round_history: list[RoundResult] = []
        self.current_round = 0

        # SCAFFOLD control variates — initialised lazily
        self._server_control: list[np.ndarray] | None = None
        self._client_controls: dict[str, list[np.ndarray]] = {}

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

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> list[np.ndarray]:
        """Initialize the global model and return its parameters."""
        self.global_model = FederatedModel.from_config(self.model_config)
        self.current_round = 0
        self.round_history = []

        # Initialise SCAFFOLD control variates to zero
        if self.strategy == AggregationStrategy.SCAFFOLD:
            self._server_control = [np.zeros_like(p) for p in self.global_model.get_parameters()]
            self._client_controls = {}

        logger.info(
            "Federation initialized — strategy=%s, config=%s",
            self.strategy.value,
            self.model_config,
        )
        return self.global_model.get_parameters()

    # ------------------------------------------------------------------
    # Global model access
    # ------------------------------------------------------------------

    def get_global_parameters(self) -> list[np.ndarray]:
        """Return current global model parameters."""
        if self.global_model is None:
            raise RuntimeError("Coordinator not initialized. Call initialize() first.")
        return self.global_model.get_parameters()

    def get_server_control(self) -> list[np.ndarray] | None:
        """Return current SCAFFOLD server control variate (or None)."""
        return self._server_control

    def get_client_control(self, client_id: str) -> list[np.ndarray]:
        """Return the stored control variate for a given client.

        Returns zeros if the client has not yet participated in a
        SCAFFOLD round.
        """
        if self._server_control is None:
            raise RuntimeError("SCAFFOLD not enabled — use strategy='scaffold'.")
        if client_id not in self._client_controls:
            self._client_controls[client_id] = [np.zeros_like(c) for c in self._server_control]
        return self._client_controls[client_id]

    # ------------------------------------------------------------------
    # Round execution
    # ------------------------------------------------------------------

    def run_round(
        self,
        client_updates: list[list[np.ndarray]],
        client_sample_counts: list[int] | None = None,
        client_ids: list[str] | None = None,
        client_control_deltas: list[list[np.ndarray]] | None = None,
        eval_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> RoundResult:
        """Execute one federation round.

        For FedAvg and FedProx this performs weighted averaging of the
        client parameter updates.  For SCAFFOLD the control-variate
        corrections are also aggregated.

        Args:
            client_updates: Parameter lists from each client.
            client_sample_counts: Samples per client for weighted averaging.
            client_ids: Client identifiers (required for SCAFFOLD).
            client_control_deltas: Per-client control variate deltas
                (SCAFFOLD only).
            eval_data: Optional ``(X, y_labels)`` for post-round eval.

        Returns:
            RoundResult with round metrics.

        Raises:
            RuntimeError: If coordinator is not initialized.
            ValueError: If too few client updates are provided.
        """
        if self.global_model is None:
            raise RuntimeError("Coordinator not initialized. Call initialize() first.")

        if len(client_updates) < self.min_clients:
            raise ValueError(
                f"Received {len(client_updates)} client updates, but minimum is {self.min_clients}."
            )

        start = time.time()
        self.current_round += 1

        # Client weights
        if client_sample_counts is not None:
            total = sum(client_sample_counts)
            weights = [c / total for c in client_sample_counts]
        else:
            weights = [1.0 / len(client_updates)] * len(client_updates)

        # Aggregate parameters
        if self.secure_aggregator is not None:
            aggregated = self.secure_aggregator.aggregate(client_updates, weights)
        else:
            aggregated = self._fedavg(client_updates, weights)

        # SCAFFOLD: aggregate control variate deltas
        if (
            self.strategy == AggregationStrategy.SCAFFOLD
            and client_control_deltas is not None
            and self._server_control is not None
        ):
            self._update_scaffold_controls(client_control_deltas, client_ids, len(client_updates))

        # Differential privacy noise
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
        converged = self._check_convergence(metrics)
        result = RoundResult(
            round_number=self.current_round,
            num_clients=len(client_updates),
            global_metrics=metrics,
            elapsed_seconds=elapsed,
            converged=converged,
        )
        self.round_history.append(result)

        logger.info(
            "Round %d complete — %d clients, metrics=%s, converged=%s",
            self.current_round,
            len(client_updates),
            metrics,
            converged,
        )
        return result

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fedavg(updates: list[list[np.ndarray]], weights: list[float]) -> list[np.ndarray]:
        """Federated Averaging: weighted mean of client parameter sets."""
        num_params = len(updates[0])
        aggregated: list[np.ndarray] = []
        for p in range(num_params):
            weighted_sum = sum(w * updates[c][p] for c, w in enumerate(weights))
            aggregated.append(weighted_sum)
        return aggregated

    def _update_scaffold_controls(
        self,
        deltas: list[list[np.ndarray]],
        client_ids: list[str] | None,
        num_clients: int,
    ) -> None:
        """Update server-level and client-level SCAFFOLD controls."""
        if self._server_control is None:
            return

        num_params = len(self._server_control)
        for p in range(num_params):
            delta_sum = sum(d[p] for d in deltas) / num_clients
            self._server_control[p] = self._server_control[p] + delta_sum

        # Store per-client controls
        if client_ids is not None:
            for i, cid in enumerate(client_ids):
                if cid not in self._client_controls:
                    self._client_controls[cid] = [np.zeros_like(c) for c in self._server_control]
                for p in range(num_params):
                    self._client_controls[cid][p] = self._client_controls[cid][p] + deltas[i][p]

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    def _check_convergence(self, metrics: dict[str, float]) -> bool:
        """Check whether training has converged.

        Uses a rolling window of accuracy values.  If the improvement
        over the window is below ``convergence_threshold`` the model is
        considered converged.
        """
        if self.convergence_window <= 0 or "accuracy" not in metrics:
            return False

        accuracies = [
            r.global_metrics.get("accuracy", 0.0)
            for r in self.round_history
            if "accuracy" in r.global_metrics
        ]
        accuracies.append(metrics["accuracy"])

        if len(accuracies) < self.convergence_window:
            return False

        window = accuracies[-self.convergence_window :]
        improvement = max(window) - min(window)
        return improvement < self.convergence_threshold

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_training_summary(self) -> dict:
        """Return a summary of all completed rounds."""
        return {
            "total_rounds": self.current_round,
            "strategy": self.strategy.value,
            "rounds": [
                {
                    "round": r.round_number,
                    "clients": r.num_clients,
                    "metrics": r.global_metrics,
                    "elapsed_s": round(r.elapsed_seconds, 4),
                    "converged": r.converged,
                }
                for r in self.round_history
            ],
        }
