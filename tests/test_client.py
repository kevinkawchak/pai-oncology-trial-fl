"""Tests for the federated client."""

import numpy as np
import pytest

from federated.client import FederatedClient, TrainingResult
from federated.model import FederatedModel, ModelConfig


@pytest.fixture()
def model_config():
    return ModelConfig(input_dim=10, hidden_dims=[8], output_dim=2, seed=42)


@pytest.fixture()
def sample_data():
    np.random.seed(42)
    x = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    return x, y


class TestFederatedClient:
    def test_init(self, model_config):
        client = FederatedClient("hospital_A", model_config)
        assert client.client_id == "hospital_A"
        assert client.model is not None

    def test_set_data(self, model_config, sample_data):
        client = FederatedClient("hospital_A", model_config)
        x, y = sample_data
        client.set_data(x, y)
        assert client.get_sample_count() == 100

    def test_train_local(self, model_config, sample_data):
        client = FederatedClient("hospital_A", model_config)
        x, y = sample_data
        client.set_data(x, y)

        global_params = FederatedModel.from_config(model_config).get_parameters()
        result = client.train_local(global_params, epochs=3, lr=0.01)

        assert isinstance(result, TrainingResult)
        assert result.client_id == "hospital_A"
        assert result.epochs_completed == 3
        assert result.num_samples == 100
        assert result.final_loss > 0

    def test_train_local_without_data_raises(self, model_config):
        client = FederatedClient("hospital_A", model_config)
        global_params = FederatedModel.from_config(model_config).get_parameters()
        with pytest.raises(RuntimeError, match="no training data"):
            client.train_local(global_params)

    def test_parameters_change_after_training(self, model_config, sample_data):
        client = FederatedClient("hospital_A", model_config)
        x, y = sample_data
        client.set_data(x, y)

        global_params = FederatedModel.from_config(model_config).get_parameters()
        client.train_local(global_params, epochs=5, lr=0.01)
        updated_params = client.get_parameters()

        # Parameters should have changed
        any_different = any(not np.allclose(g, u) for g, u in zip(global_params, updated_params))
        assert any_different

    def test_update_model(self, model_config):
        client = FederatedClient("hospital_A", model_config)
        new_params = [np.ones_like(p) for p in client.get_parameters()]
        client.update_model(new_params)
        for p in client.get_parameters():
            assert np.allclose(p, np.ones_like(p))
