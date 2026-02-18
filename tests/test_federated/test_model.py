"""Tests for federated/model.py — FederatedModel and ModelConfig.

Covers creation, forward pass shapes, get/set parameters, parameter
count, training steps, evaluation, softmax properties, and edge cases.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from tests.conftest import PROJECT_ROOT, load_module

mod = load_module("federated.model", "federated/model.py")

FederatedModel = mod.FederatedModel
ModelConfig = mod.ModelConfig


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_config(self):
        cfg = ModelConfig()
        assert cfg.input_dim == 30
        assert cfg.hidden_dims == [64, 32]
        assert cfg.output_dim == 2
        assert cfg.learning_rate == 0.01
        assert cfg.seed == 42

    def test_custom_config(self):
        cfg = ModelConfig(input_dim=10, hidden_dims=[16], output_dim=3, seed=99)
        assert cfg.input_dim == 10
        assert cfg.hidden_dims == [16]
        assert cfg.output_dim == 3


class TestFederatedModelCreation:
    """Tests for model instantiation."""

    def test_creation_defaults(self):
        m = FederatedModel()
        assert m.input_dim == 30
        assert m.output_dim == 2
        assert len(m.layers) == 3  # 30->64, 64->32, 32->2

    def test_creation_custom_dims(self):
        m = FederatedModel(input_dim=10, hidden_dims=[16, 8], output_dim=3, seed=0)
        assert len(m.layers) == 3
        assert m.layers[0][0].shape == (10, 16)
        assert m.layers[1][0].shape == (16, 8)
        assert m.layers[2][0].shape == (8, 3)

    def test_from_config(self):
        cfg = ModelConfig(input_dim=5, hidden_dims=[10], output_dim=2, seed=7)
        m = FederatedModel.from_config(cfg)
        assert m.input_dim == 5
        assert m.output_dim == 2
        assert len(m.layers) == 2  # 5->10, 10->2

    def test_single_hidden_layer(self):
        m = FederatedModel(input_dim=4, hidden_dims=[8], output_dim=2, seed=0)
        assert len(m.layers) == 2

    def test_no_hidden_layers(self):
        m = FederatedModel(input_dim=4, hidden_dims=[], output_dim=2, seed=0)
        assert len(m.layers) == 1  # direct: 4->2


class TestForwardPass:
    """Tests for forward pass behavior and output shapes."""

    def test_forward_output_shape(self):
        m = FederatedModel(input_dim=4, hidden_dims=[8], output_dim=3, seed=0)
        x = np.random.randn(10, 4).astype(np.float32)
        out = m.forward(x)
        assert out.shape == (10, 3)

    def test_forward_produces_probabilities(self):
        m = FederatedModel(input_dim=4, hidden_dims=[8], output_dim=3, seed=0)
        x = np.random.randn(5, 4)
        out = m.forward(x)
        # Each row sums to 1
        npt.assert_allclose(out.sum(axis=1), np.ones(5), atol=1e-6)
        # All values non-negative
        assert np.all(out >= 0)

    def test_forward_single_sample(self):
        m = FederatedModel(input_dim=4, hidden_dims=[8], output_dim=2, seed=0)
        x = np.random.randn(1, 4)
        out = m.forward(x)
        assert out.shape == (1, 2)
        npt.assert_allclose(out.sum(), 1.0, atol=1e-6)

    def test_predict_returns_class_indices(self):
        m = FederatedModel(input_dim=4, hidden_dims=[8], output_dim=3, seed=0)
        x = np.random.randn(10, 4)
        preds = m.predict(x)
        assert preds.shape == (10,)
        assert all(0 <= p < 3 for p in preds)


class TestGetSetParameters:
    """Tests for parameter extraction and injection."""

    def test_get_parameters_count(self):
        m = FederatedModel(input_dim=4, hidden_dims=[8, 4], output_dim=2, seed=0)
        params = m.get_parameters()
        # 3 layers => 6 arrays (W, b each)
        assert len(params) == 6

    def test_get_parameters_returns_copies(self):
        m = FederatedModel(input_dim=4, hidden_dims=[8], output_dim=2, seed=0)
        params = m.get_parameters()
        params[0][0, 0] = 999.0
        fresh = m.get_parameters()
        assert fresh[0][0, 0] != 999.0

    def test_set_parameters_changes_model(self):
        m = FederatedModel(input_dim=4, hidden_dims=[8], output_dim=2, seed=0)
        new_params = [np.ones_like(p) for p in m.get_parameters()]
        m.set_parameters(new_params)
        params = m.get_parameters()
        for p in params:
            npt.assert_allclose(p, np.ones_like(p))

    def test_set_parameters_uses_copies(self):
        m = FederatedModel(input_dim=4, hidden_dims=[8], output_dim=2, seed=0)
        new_params = [np.ones_like(p) for p in m.get_parameters()]
        m.set_parameters(new_params)
        new_params[0][0, 0] = 999.0
        current = m.get_parameters()
        assert current[0][0, 0] != 999.0

    def test_total_parameter_count(self):
        m = FederatedModel(input_dim=4, hidden_dims=[8], output_dim=2, seed=0)
        params = m.get_parameters()
        total = sum(p.size for p in params)
        # W1: 4*8=32, b1: 8, W2: 8*2=16, b2: 2 => 58
        assert total == 58


class TestTrainAndEvaluate:
    """Tests for training step and evaluation."""

    def test_train_step_returns_loss(self):
        m = FederatedModel(input_dim=4, hidden_dims=[8], output_dim=2, seed=0)
        x = np.random.randn(10, 4)
        y_onehot = np.zeros((10, 2))
        y_onehot[np.arange(10), np.random.randint(0, 2, 10)] = 1.0
        loss = m.train_step(x, y_onehot, lr=0.01)
        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate_returns_metrics(self):
        m = FederatedModel(input_dim=4, hidden_dims=[8], output_dim=2, seed=0)
        x = np.random.randn(20, 4)
        y = np.random.randint(0, 2, 20)
        metrics = m.evaluate(x, y)
        assert "accuracy" in metrics
        assert "loss" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["loss"] > 0
