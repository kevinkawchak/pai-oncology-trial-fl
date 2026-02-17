"""Federated client — simulates a hospital or research-center node.

Each client holds a local dataset partition and trains the shared model
architecture on its own data.  Only model-parameter updates (never raw
patient data) leave the client.

Supports three training modes that correspond to the coordinator's
aggregation strategy:

- **FedAvg** — standard local SGD.
- **FedProx** — adds a proximal regularisation term
  ``(mu/2) * ||w - w_global||^2`` to the loss function, reducing
  client drift under non-IID data.
- **SCAFFOLD** — applies server and client control variates to correct
  gradient updates, improving convergence in heterogeneous settings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from federated.model import FederatedModel, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Outcome of a local training run at one client."""

    client_id: str
    epochs_completed: int
    final_loss: float
    num_samples: int
    metrics: dict[str, float] = field(default_factory=dict)


class FederatedClient:
    """Simulated hospital node for federated oncology training.

    Each client:
    1. Receives global model parameters from the coordinator.
    2. Trains locally on its private data partition.
    3. Returns updated parameters (never raw data).

    Args:
        client_id: Unique identifier (e.g. ``"hospital_A"``).
        model_config: Architecture specification (must match the coordinator).
        x_train: Local training features.
        y_train: Local training labels (integer class indices).
    """

    def __init__(
        self,
        client_id: str,
        model_config: ModelConfig,
        x_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
    ):
        self.client_id = client_id
        self.model_config = model_config
        self.model = FederatedModel.from_config(model_config)
        self.x_train = x_train
        self.y_train = y_train
        self._training_history: list[TrainingResult] = []

    def set_data(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Load a data partition into this client."""
        self.x_train = x_train
        self.y_train = y_train

    def update_model(self, global_params: list[np.ndarray]) -> None:
        """Replace local model parameters with the global model."""
        self.model.set_parameters(global_params)

    # ------------------------------------------------------------------
    # Standard FedAvg / FedProx training
    # ------------------------------------------------------------------

    def train_local(
        self,
        global_params: list[np.ndarray],
        epochs: int = 5,
        lr: float | None = None,
        batch_size: int = 32,
        mu: float = 0.0,
    ) -> TrainingResult:
        """Train locally and return updated parameters.

        When ``mu > 0`` a FedProx proximal term is added to the loss so
        that the local model stays close to the global model.  This
        improves convergence under heterogeneous (non-IID) data.

        Args:
            global_params: Current global model parameters.
            epochs: Number of local epochs.
            lr: Learning rate (defaults to model_config.learning_rate).
            batch_size: Mini-batch size.
            mu: FedProx proximal coefficient.  ``0`` disables the term.

        Returns:
            TrainingResult with loss and sample count.

        Raises:
            RuntimeError: If no training data has been loaded.
        """
        if self.x_train is None or self.y_train is None:
            raise RuntimeError(f"Client {self.client_id}: no training data loaded.")

        if lr is None:
            lr = self.model_config.learning_rate

        self.model.set_parameters(global_params)
        n_samples = len(self.x_train)
        n_classes = self.model_config.output_dim

        # One-hot encode labels
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), self.y_train] = 1.0

        epoch_loss = 0.0
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = indices[start:end]
                loss = self.model.train_step(self.x_train[idx], y_onehot[idx], lr)

                # FedProx proximal term: penalise deviation from global
                if mu > 0:
                    local_params = self.model.get_parameters()
                    for i, (lp, gp) in enumerate(zip(local_params, global_params)):
                        local_params[i] = lp - lr * mu * (lp - gp)
                    self.model.set_parameters(local_params)

                epoch_loss += loss
                batches += 1
            epoch_loss /= max(batches, 1)

        metrics = self.model.evaluate(self.x_train, self.y_train)
        result = TrainingResult(
            client_id=self.client_id,
            epochs_completed=epochs,
            final_loss=epoch_loss,
            num_samples=n_samples,
            metrics=metrics,
        )
        self._training_history.append(result)
        logger.info(
            "Client %s trained for %d epochs — loss=%.4f, acc=%.4f",
            self.client_id,
            epochs,
            epoch_loss,
            metrics.get("accuracy", 0.0),
        )
        return result

    # ------------------------------------------------------------------
    # SCAFFOLD training
    # ------------------------------------------------------------------

    def train_local_scaffold(
        self,
        global_params: list[np.ndarray],
        server_control: list[np.ndarray],
        client_control: list[np.ndarray],
        epochs: int = 5,
        lr: float | None = None,
        batch_size: int = 32,
    ) -> tuple[TrainingResult, list[np.ndarray]]:
        """Train locally using the SCAFFOLD algorithm.

        After local SGD each gradient step is corrected by
        ``server_control - client_control``.  The method also returns
        the control-variate delta used to update the server control.

        Args:
            global_params: Current global model parameters.
            server_control: Server-level control variate ``c``.
            client_control: This client's control variate ``c_i``.
            epochs: Number of local epochs.
            lr: Learning rate.
            batch_size: Mini-batch size.

        Returns:
            Tuple of (TrainingResult, control_variate_delta).
        """
        if self.x_train is None or self.y_train is None:
            raise RuntimeError(f"Client {self.client_id}: no training data loaded.")

        if lr is None:
            lr = self.model_config.learning_rate

        self.model.set_parameters(global_params)
        n_samples = len(self.x_train)
        n_classes = self.model_config.output_dim
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), self.y_train] = 1.0

        total_steps = 0
        epoch_loss = 0.0
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = indices[start:end]
                loss = self.model.train_step(self.x_train[idx], y_onehot[idx], lr)

                # SCAFFOLD correction: w -= lr * (c - c_i)
                local_params = self.model.get_parameters()
                for i, (sc, cc) in enumerate(zip(server_control, client_control)):
                    local_params[i] = local_params[i] - lr * (sc - cc)
                self.model.set_parameters(local_params)

                epoch_loss += loss
                batches += 1
                total_steps += 1
            epoch_loss /= max(batches, 1)

        # Compute control variate delta: delta_c_i = (1 / (K * lr)) * (x_global - x_local) - c
        new_params = self.model.get_parameters()
        k_lr = max(total_steps, 1) * lr
        control_delta = []
        for gp, lp, cc in zip(global_params, new_params, client_control):
            delta = (gp - lp) / k_lr - cc
            control_delta.append(delta)

        metrics = self.model.evaluate(self.x_train, self.y_train)
        result = TrainingResult(
            client_id=self.client_id,
            epochs_completed=epochs,
            final_loss=epoch_loss,
            num_samples=n_samples,
            metrics=metrics,
        )
        self._training_history.append(result)
        logger.info(
            "Client %s SCAFFOLD trained for %d epochs — loss=%.4f",
            self.client_id,
            epochs,
            epoch_loss,
        )
        return result, control_delta

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_parameters(self) -> list[np.ndarray]:
        """Return current local model parameters."""
        return self.model.get_parameters()

    def get_sample_count(self) -> int:
        """Number of local training samples."""
        if self.x_train is None:
            return 0
        return len(self.x_train)

    def get_training_history(self) -> list[TrainingResult]:
        """Return all past training results for this client."""
        return list(self._training_history)
