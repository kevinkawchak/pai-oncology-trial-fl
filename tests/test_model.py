"""Tests for the federated model (numpy MLP)."""

import numpy as np

from federated.model import FederatedModel, ModelConfig


class TestModelConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.input_dim == 30
        assert config.hidden_dims == [64, 32]
        assert config.output_dim == 2

    def test_custom_config(self):
        config = ModelConfig(input_dim=10, hidden_dims=[16], output_dim=3)
        assert config.input_dim == 10
        assert config.output_dim == 3


class TestFederatedModel:
    def test_init(self):
        model = FederatedModel(input_dim=10, hidden_dims=[8], output_dim=2)
        assert len(model.layers) == 2  # input->hidden, hidden->output

    def test_from_config(self):
        config = ModelConfig(input_dim=10, hidden_dims=[8], output_dim=2)
        model = FederatedModel.from_config(config)
        assert model.input_dim == 10
        assert model.output_dim == 2

    def test_forward_shape(self):
        model = FederatedModel(input_dim=10, hidden_dims=[8], output_dim=3)
        x = np.random.randn(5, 10)
        out = model.forward(x)
        assert out.shape == (5, 3)

    def test_forward_probabilities(self):
        model = FederatedModel(input_dim=10, hidden_dims=[8], output_dim=3)
        x = np.random.randn(5, 10)
        out = model.forward(x)
        # Outputs should be valid probabilities
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(out >= 0)

    def test_predict(self):
        model = FederatedModel(input_dim=10, hidden_dims=[8], output_dim=2)
        x = np.random.randn(5, 10)
        preds = model.predict(x)
        assert preds.shape == (5,)
        assert all(p in (0, 1) for p in preds)

    def test_get_set_parameters(self):
        model = FederatedModel(input_dim=10, hidden_dims=[8], output_dim=2, seed=0)
        params = model.get_parameters()
        assert len(params) == 4  # 2 layers * (weight + bias)

        # Modify and set back
        model2 = FederatedModel(input_dim=10, hidden_dims=[8], output_dim=2, seed=1)
        model2.set_parameters(params)
        params2 = model2.get_parameters()
        for p1, p2 in zip(params, params2):
            assert np.allclose(p1, p2)

    def test_train_step_reduces_loss(self):
        np.random.seed(42)
        model = FederatedModel(input_dim=10, hidden_dims=[16], output_dim=2, seed=42)
        x = np.random.randn(50, 10)
        y = np.zeros((50, 2))
        y[np.arange(50), np.random.randint(0, 2, 50)] = 1.0

        loss1 = model.train_step(x, y, lr=0.01)
        for _ in range(20):
            loss2 = model.train_step(x, y, lr=0.01)
        assert loss2 < loss1

    def test_evaluate(self):
        model = FederatedModel(input_dim=10, hidden_dims=[8], output_dim=2)
        x = np.random.randn(20, 10)
        y = np.random.randint(0, 2, 20)
        metrics = model.evaluate(x, y)
        assert "accuracy" in metrics
        assert "loss" in metrics
        assert 0 <= metrics["accuracy"] <= 1


class TestReproducibility:
    def test_same_seed_same_model(self):
        m1 = FederatedModel(input_dim=5, hidden_dims=[4], output_dim=2, seed=99)
        m2 = FederatedModel(input_dim=5, hidden_dims=[4], output_dim=2, seed=99)
        for p1, p2 in zip(m1.get_parameters(), m2.get_parameters()):
            assert np.allclose(p1, p2)

    def test_different_seed_different_model(self):
        m1 = FederatedModel(input_dim=5, hidden_dims=[4], output_dim=2, seed=1)
        m2 = FederatedModel(input_dim=5, hidden_dims=[4], output_dim=2, seed=2)
        any_different = any(not np.allclose(p1, p2) for p1, p2 in zip(m1.get_parameters(), m2.get_parameters()))
        assert any_different
