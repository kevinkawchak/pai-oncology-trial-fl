"""Neural network model for federated oncology trials.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

Provides a numpy-based multilayer perceptron that can be trained
across federated sites without requiring PyTorch or TensorFlow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelConfig:
    """Configuration for a federated model architecture.

    Attributes:
        input_dim: Number of input features.
        hidden_dims: List of hidden layer sizes.
        output_dim: Number of output classes.
        learning_rate: Default learning rate for training.
        seed: Random seed for reproducibility.
    """

    input_dim: int = 30
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 2
    learning_rate: float = 0.01
    seed: int | None = 42


class FederatedModel:
    """Numpy-based multilayer perceptron for federated learning.

    Supports forward/backward passes, parameter extraction/injection,
    and local training — all operations needed for federated averaging.

    Args:
        input_dim: Number of input features.
        hidden_dims: List of hidden layer sizes.
        output_dim: Number of output classes.
        seed: Random seed for weight initialization.
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dims: list[int] | None = None,
        output_dim: int = 2,
        seed: int | None = 42,
    ):
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        rng = np.random.default_rng(seed)
        dims = [input_dim, *hidden_dims, output_dim]
        self.layers: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            w = rng.normal(0, scale, (dims[i], dims[i + 1]))
            b = np.zeros(dims[i + 1])
            self.layers.append((w, b))

    @classmethod
    def from_config(cls, config: ModelConfig) -> FederatedModel:
        """Create a model from a ModelConfig."""
        return cls(
            input_dim=config.input_dim,
            hidden_dims=list(config.hidden_dims),
            output_dim=config.output_dim,
            seed=config.seed,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network.

        Args:
            x: Input array of shape (batch_size, input_dim).

        Returns:
            Softmax probabilities of shape (batch_size, output_dim).
        """
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.maximum(0, x)  # ReLU
        return _softmax(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels for input samples."""
        probs = self.forward(x)
        return np.argmax(probs, axis=1)

    def get_parameters(self) -> list[np.ndarray]:
        """Extract all model parameters as a flat list of numpy arrays."""
        params = []
        for w, b in self.layers:
            params.extend([w.copy(), b.copy()])
        return params

    def set_parameters(self, params: list[np.ndarray]) -> None:
        """Set model parameters from a flat list of numpy arrays."""
        for i in range(len(self.layers)):
            w = params[2 * i]
            b = params[2 * i + 1]
            self.layers[i] = (w.copy(), b.copy())

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """Perform one training step with backpropagation.

        Args:
            x: Input features of shape (batch_size, input_dim).
            y: One-hot encoded labels of shape (batch_size, output_dim).
            lr: Learning rate.

        Returns:
            Cross-entropy loss for this batch.
        """
        # Forward pass — cache activations
        activations = [x]
        pre_activations: list[np.ndarray] = []
        for i, (w, b) in enumerate(self.layers):
            z = activations[-1] @ w + b
            pre_activations.append(z)
            if i < len(self.layers) - 1:
                a = np.maximum(0, z)
            else:
                a = _softmax(z)
            activations.append(a)

        # Backward pass
        n = x.shape[0]
        delta = activations[-1] - y  # softmax + cross-entropy gradient

        for i in range(len(self.layers) - 1, -1, -1):
            w, b = self.layers[i]
            dw = activations[i].T @ delta / n
            db = np.mean(delta, axis=0)

            if i > 0:
                delta = delta @ w.T
                delta = delta * (pre_activations[i - 1] > 0).astype(float)

            self.layers[i] = (w - lr * dw, b - lr * db)

        # Compute loss
        probs = np.clip(activations[-1], 1e-10, 1.0)
        loss = -np.mean(np.sum(y * np.log(probs), axis=1))
        return float(loss)

    def evaluate(self, x: np.ndarray, y_labels: np.ndarray) -> dict[str, float]:
        """Evaluate model accuracy and loss.

        Args:
            x: Input features.
            y_labels: Integer class labels.

        Returns:
            Dictionary with 'accuracy' and 'loss' keys.
        """
        probs = self.forward(x)
        predictions = np.argmax(probs, axis=1)
        accuracy = float(np.mean(predictions == y_labels))

        y_onehot = np.zeros((len(y_labels), self.output_dim))
        y_onehot[np.arange(len(y_labels)), y_labels] = 1.0
        probs_clipped = np.clip(probs, 1e-10, 1.0)
        loss = -float(np.mean(np.sum(y_onehot * np.log(probs_clipped), axis=1)))

        return {"accuracy": accuracy, "loss": loss}


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
